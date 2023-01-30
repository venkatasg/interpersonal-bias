'''
Script to label tweet with interpersonal emotions
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
from train_emot_labeller import multilabel_f1, emo_labeller
import ipdb
import argparse

# Logging level for datasets
loggingd.disable_progress_bar()

 # Set logging to show only errors for transformers
loggingt.set_verbosity_error()

# Turn off chained assignment warnings from pandas
pd.set_option('chained_assignment',None)


if __name__== "__main__":
    # initialize argument parser
    description = 'Script to label tweet with interpersonal emotions'
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
    df['labels'] = df.apply(lambda x: x[emot_ans].values.astype(float), axis=1)
    
    ## Find weighting for the actual classification loss
    support = np.sum(df.loc[:, emot_ans].values, axis=0)
    total = np.sum(support)
    not_support = np.array([(total-x) for x in support])
    weights = support/not_support
    
    # Initialise model, then load weights
    model = emo_labeller(
        lm_model='vinai/bertweet-base',
        n_classes=len(emot_ans),
        weights=weights
    )
    model.load_state_dict(torch.load(args.model_path + '/model.bin'))
    
    # Load the AutoTokenizer with a normalization mode if the input Tweet is raw
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, normalization=True)

    # Preprocess and get data ready
    data = Dataset.from_pandas(df)
    data = data.map(lambda x: tokenizer(x['tweet'], truncation=True, padding="max_length", max_length=128), batched=True)
        
    if args.dev:
        test_df = df[df['Split']=='dev']
        test_data = data.filter(lambda x: x['Split']=='dev')
    else:
        test_df = df[df['Split']=='test']
        test_data = data.filter(lambda x: x['Split']=='test')
    
    test_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch, shuffle=False)
    
    test_dataloader, model = accelerator.prepare(test_dataloader, model)

    model.zero_grad()
    model.eval()
     
    # Tracking variables
    predictions = None
    references = None
    
    for _, input_dict in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model(**input_dict)
        
        if predictions is None:
            predictions = outputs['preds'].detach().cpu().numpy()
            references = input_dict['labels'].int().detach().cpu().numpy()
        else:
            predictions = np.append(predictions, outputs['preds'].detach().cpu().numpy(), axis=0)
            references = np.append(references, input_dict['labels'].int().detach().cpu().numpy(), axis=0)
    
    f1_metric = multilabel_f1(predictions, references)
    print("F1 Score:", np.round(f1_metric['micro_f1']*100, 1))
    for i, emot in enumerate(emot_ans):
        print(emot + ":" + str(np.round(f1_metric['individual_f1s'][i]*100,1)), end=', ')
    
    true_zero = [np.all(x==0) for x in references]
    pred_zero = [np.all(x==0) for x in predictions]
    null_f1 = f1_score(true_zero, pred_zero, average='micro', zero_division=0)
    print("\nNull emotions F1 score:", np.round(null_f1*100, 1))
    
    test_df.loc[:, 'pred'] = predictions.tolist()
    
    in_inds = np.where(test_df['group'].values==1)[0]
    f1in = multilabel_f1(references[in_inds], predictions[in_inds])
    print("In-group F1-scores:", np.round(f1in['micro_f1']*100, 1))
    for i, emot in enumerate(emot_ans):
        print(emot + ":" + str(np.round(f1in['individual_f1s'][i]*100,1)), end=', ')

    out_inds = np.where(test_df['group'].values==0)[0]
    f1out = multilabel_f1(references[out_inds], predictions[out_inds])
    print("\nOut-group F1-scores:", np.round(f1out['micro_f1']*100, 1))
    for i, emot in enumerate(emot_ans):
        print(emot + ":" + str(np.round(f1out['individual_f1s'][i]*100,1)), end=', ')
    
    print("\n")
    
    if args.dev:
        test_df.to_csv('predictions-emot.tsv', sep='\t', index=False)
