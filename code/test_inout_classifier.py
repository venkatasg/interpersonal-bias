'''
Script to classify tweet as in-group or out-group on dev or test set
'''
import torch
import pandas as pd
import random
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
    set_seed,
    logging as loggingt
)
from accelerate import Accelerator
from datasets import Dataset, logging as loggingd
import evaluate
from sklearn.metrics import f1_score
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
    description = 'Script to classify tweets as in-group or out-group'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '--batch',
        type=int,
        default=32,
        help='Batch size for training.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed.'
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

    # Initialise model and load weights
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    # Load the AutoTokenizer with a normalization mode
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, normalization=True)

    df = pd.read_csv(args.data_path, sep='\t')
    data = Dataset.from_pandas(df)
    data = data.map(lambda x: tokenizer(x['tweet'], truncation=True, padding="max_length", max_length=128), batched=True)
    # Rename group column to labels for use with BERT model
    data = data.rename_columns({'group': 'labels'})

    if args.dev:
        test_df = df[df['Split']=='dev']
        test_data = data.filter(lambda x: x['Split']=='dev')
    else:
        test_df = df[df['Split']=='test']
        test_data = data.filter(lambda x: x['Split']=='test')

    test_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch, shuffle=False)
    
    test_dataloader, model = accelerator.prepare(test_dataloader, model)
    
    # Load evaluation metric
    f1_metric = evaluate.load("f1")

    model.zero_grad()
    model.eval()
    preds = np.array([])
    for _, input_dict in enumerate(test_dataloader):
        with torch.no_grad():
            output = model(**input_dict)
            
        predictions = output['logits'].argmax(dim=1)
        
        # Store predictions and references for evalation later
        predictions, references = accelerator.gather_for_metrics((predictions, input_dict['labels']))
        preds = np.append(preds, predictions.detach().cpu().numpy())
        f1_metric.add_batch(
            predictions=predictions,
            references=references,
        )

    print("F1 Score: ", np.round(f1_metric.compute(average='micro')['f1']*100, 1))
    
    test_df.loc[:, 'pred'] = preds

    for emot in ['Admiration', 'Anger', 'Disgust', 'Joy', 'Sadness', 'Interest', 'Fear', 'Surprise']:
        f1s = f1_score(test_df[test_df[emot]==True]['group'], test_df[test_df[emot]==True]['pred'], average='micro', zero_division=0)
        print(emot + ': ' + str(np.round(f1s*100, 1)), end=', ')

    r_f1 = f1_score(test_df[test_df['party']=='R']['group'], test_df[test_df['party']=='R']['pred'], average='micro')
    print('\nF1 score on republican tweets: ', np.round(r_f1*100, 1))

    d_f1 = f1_score(test_df[test_df['party']=='D']['group'], test_df[test_df['party']=='D']['pred'], average='micro')
    print('F1 score on democrat tweets: ', np.round(d_f1*100, 1))
    
    print("\n")
    
    if args.dev:
        test_df.to_csv('predictions-inout.tsv', sep='\t', index=False)
