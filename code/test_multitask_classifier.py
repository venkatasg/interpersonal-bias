'''
Multitask testing of inout group and emotions
'''
import torch
import pandas as pd
import random
import numpy as np
from transformers import (
    AutoModel,
    AutoTokenizer,
    set_seed,
    logging as loggingt
)
from accelerate import Accelerator
from datasets import Dataset, logging as loggingd
from sklearn.metrics import f1_score
import ipdb
import argparse
from train_multitask_classifier import MultiTaskModel
from train_emot_labeller import multilabel_f1

# Logging level for datasets
loggingd.disable_progress_bar()

 # Set logging to show only errors for transformers
loggingt.set_verbosity_error()

# Turn off chained assignment warnings from pandas
pd.set_option('chained_assignment',None)

if __name__== "__main__":
    # initialize argument parser
    description = 'Script to classify sentences as ingroup and outgroup, as well as emotions at same time'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '--batch',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed'
    )
    parser.add_argument(
        '--dev',
        action='store_true',
        help='test on dev set.'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to data to test on.'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to saved model to use.'
    )
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)
    random.seed(args.seed)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()

    emot_ans = ['Admiration', 'Anger', 'Disgust', 'Fear', 'Interest', 'Joy', 'Sadness', 'Surprise']
    
    # Loading the data
    df = pd.read_csv(args.data_path, sep='\t')
    # Cast emot labels as floats
    df['emot'] = df.apply(lambda x: x[emot_ans].values.astype(float), axis=1)
    
    # Load the AutoTokenizer with a normalization mode
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, normalization=True)

    # df = df.drop(df[df['Split'] == 'train'].sample(frac=.5).index)
    data = Dataset.from_pandas(df)
    data = data.map(lambda x: tokenizer(x['tweet'], truncation=True, padding="max_length", max_length=128), batched=True)

    if args.dev:
        test_df = df[df['Split']=='dev']
        test_data = data.filter(lambda x: x['Split']=='dev')
    else:
        test_df = df[df['Split']=='test']
        test_data = data.filter(lambda x: x['Split']=='test')
    test_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'group', 'emot'])

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch, shuffle=False)
    
    # Define model then load weights
    model = MultiTaskModel(
        n_classes=len(emot_ans),
        lm_model='vinai/bertweet-base',
        weights=torch.ones(len(emot_ans))    # This isn't used, but it prevents error when loading the state dict
    )
    model.load_state_dict(torch.load(args.model_path + '/model.bin'))

    model.zero_grad()
    model.eval()
    
    test_dataloader, model = accelerator.prepare(test_dataloader, model)
    
    # Tracking variables
    predictions_group = np.array([])
    references_group = np.array([])
    predictions_emot = None
    references_emot = None
    
    for _, input_dict in enumerate(test_dataloader):
        for task in ['group', 'emot']:
            with torch.no_grad():
                test_dict = {
                    'input_ids': input_dict['input_ids'],
                    'attention_mask': input_dict['attention_mask'],
                    'task': task
                }
                outputs = model(**test_dict)
                
            if task=='emot':
                if predictions_emot is None:
                    predictions_emot = outputs['preds'].detach().cpu().numpy()
                    references_emot = input_dict['emot'].int().detach().cpu().numpy()
                else:
                    predictions_emot = np.append(predictions_emot, outputs['preds'].detach().cpu().numpy(), axis=0)
                    references_emot = np.append(references_emot, input_dict['emot'].int().detach().cpu().numpy(), axis=0)
            else:
                predictions_group = np.append(predictions_group, outputs['preds'].detach().cpu().numpy())
                references_group = np.append(references_group, input_dict['group'].detach().cpu().numpy())
                
    f1_emot = multilabel_f1(references_emot, predictions_emot)
    f1_group = multilabel_f1(references_group, predictions_group)
    
    print('Group F1 Score:', np.round(f1_group['micro_f1']*100, 1))
    print('Emo F1 Score:', np.round(f1_emot['micro_f1']*100, 1))
    for i, emot in enumerate(emot_ans):
        print(emot + ":" + str(np.round(f1_emot['individual_f1s'][i]*100,1)), end=', ')

    true_zero = [np.all(x==0) for x in references_emot]
    pred_zero = [np.all(x==0) for x in predictions_emot]
    null_f1 = f1_score(true_zero, pred_zero, average='micro', zero_division=0)
    print("\nNull emotions F1 score:", np.round(null_f1*100, 3))

    test_df.loc[:, 'group_pred'] = predictions_group
    test_df.loc[:, 'emot_pred'] = predictions_emot.tolist()

    in_inds = np.where(references_group==1)[0]
    f1in_emot = multilabel_f1(references_emot[in_inds], predictions_emot[in_inds])
    print("In-group F1-scores:", np.round(f1in_emot['micro_f1']*100, 1))
    for i, emot in enumerate(emot_ans):
        print(emot + ":" + str(np.round(f1in_emot['individual_f1s'][i]*100,1)), end=', ')

    out_inds = np.where(references_group==0)[0]
    f1out_emot = multilabel_f1(references_emot[out_inds], predictions_emot[out_inds])
    print("\nOut-group F1-scores:", np.round(f1out_emot['micro_f1']*100, 1))
    for i, emot in enumerate(emot_ans):
        print(emot + ":" + str(np.round(f1out_emot['individual_f1s'][i]*100,1)), end=', ')
    
    print("\n")
    
    if args.dev:
        test_df.to_csv('predictions-multitask.tsv', sep='\t', index=False)



