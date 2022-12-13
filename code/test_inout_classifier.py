'''
Script to classify in-group or out-group on dev or test set
'''
import torch
import pandas as pd
import random
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler, logging as loggingt
from datasets import Dataset
from datasets import logging as loggingd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score as acc, precision_score as prec, recall_score as rec
import ipdb
import argparse
import re
import gc
import sys

# Logging level for datasets
loggingd.disable_progress_bar()

 # Set logging to show only errors for transformers
loggingt.set_verbosity_error()

# Turn off chained assignment warnings from pandas
pd.set_option('chained_assignment',None)

def replace_ent(tweet, ent):
    'Find entity name in tweet and replace with placeholder'

    pattern = re.compile(r"\@" + ent, re.IGNORECASE)
    return re.sub(pattern, "@USER", tweet)

if __name__== "__main__":
    # initialize argument parser
    description = 'Script to classify sentences as in-group or out-group'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '--batch',
        type=int,
        default=64,
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
        help='test on dev set'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to data'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to saved model to use'
    )

    args = parser.parse_args()

    # Free up memory
    gc.collect()
    torch.cuda.empty_cache()

    # Choose gpu or cpu
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(device)

    # Set random seeds for reproducibility on a specific machine
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    np.random.RandomState(args.seed)

    # Initialise model and load weights
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    # Load the AutoTokenizer with a normalization mode if the input Tweet is raw
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    df = pd.read_csv(args.data_path, sep='\t')
    df['tweet_clean'] = df.apply(lambda x: replace_ent(tweet=x['tweet'], ent=x['mentname']), axis=1)
    df['group'] = df['group'].apply(lambda x: 0 if x==-1 else 1)


    data = Dataset.from_pandas(df)
    data = data.map(lambda x: tokenizer(x['tweet'], truncation=True, padding="max_length", max_length=128), batched=True)

    if args.dev:
        test_df = df[df['Split']=='dev']
        test_data = data.filter(lambda x: x['Split']=='dev')
    else:
        test_df = df[df['Split']=='test']
        test_data = data.filter(lambda x: x['Split']=='test')

    test_data.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'group'])

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch, shuffle=False)

    model.zero_grad()
    model.eval()

     # Tracking variables
    pred_labels = np.array([])
    true_labels = np.array([])

    with torch.no_grad():
        for _, input_dict in enumerate(test_dataloader):
            b_input_ids = input_dict['input_ids'].to(device)
            b_input_mask = input_dict['attention_mask'].to(device)

            out = model(input_ids=b_input_ids, attention_mask=b_input_mask)

            logits = out['logits'].detach().cpu().numpy()
            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat= input_dict['group'].to('cpu').numpy().flatten()

            pred_labels = np.append(pred_labels, pred_flat)
            true_labels = np.append(true_labels, labels_flat)

    test_df.loc[:, 'pred'] = pred_labels.astype(int)

    fscore = f1_score(true_labels, pred_labels, average='micro')
    print('\nF1 Score:', np.round(fscore,3))

    for emot in ['Admiration', 'Anger', 'Disgust', 'Joy', 'Sadness', 'Interest', 'Fear', 'Surprise']:
        f1s = f1_score(test_df[test_df[emot]==True]['group'], test_df[test_df[emot]==True]['pred'], average='micro', zero_division=0)
        print(emot + ' F1 score:', np.round(f1s, 3))

    r_f1 = f1_score(test_df[test_df['party']=='R']['group'], test_df[test_df['party']=='R']['pred'], average='micro')
    print('F1 score on republican tweets', np.round(r_f1, 3))

    d_f1 = f1_score(test_df[test_df['party']=='D']['group'], test_df[test_df['party']=='D']['pred'], average='micro')
    print('F1 score on democrat tweets', np.round(d_f1, 3))

    print("===================================================")

    if args.dev:
        test_df.to_csv('predictions.tsv', sep='\t', index=False)
