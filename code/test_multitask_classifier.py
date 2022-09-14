'''
Multitask testing of inout group and emotions
'''
import torch
import pandas as pd
import random
import numpy as np
from transformers import AutoModel, AutoTokenizer, get_scheduler, logging as loggingt
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
import os

# Logging level for datasets
loggingd.disable_progress_bar()

 # Set logging to show only errors for transformers
loggingt.set_verbosity_error()

def replace_ent(tweet, ent):
    'Find entity name in tweet and replace with placeholder' 
    
    pattern = re.compile(r"\@" + ent, re.IGNORECASE)
    return re.sub(pattern, "@USER", tweet)


class MultiTaskModel(torch.nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        self.encoder = AutoModel.from_pretrained("vinai/bertweet-base")
        
        # emot labelling heads
        self.dropout_emot = torch.nn.Dropout(0.1)
        self.pooler_emot = torch.nn.Linear(768, 8)
        
        # inout classification head
        self.dropout_inout = torch.nn.Dropout(0.1)
        self.pooler_inout = torch.nn.Linear(768,2)

    def forward(self, input_ids, attention_mask, task):
        
        # Take output from <s> token
        outputs = self.encoder(input_ids, attention_mask)
        
        # Do task specific heads
        if task=='classification':
            out_1 = self.dropout_inout(outputs['pooler_output'])
            out_2 = self.pooler_inout(out_1)
        else:
            out_1 = self.dropout_emot(outputs['pooler_output'])
            out_2 = self.pooler_emot(out_1)            
        
        return out_2

if __name__== "__main__":
    # initialize argument parser
    description = 'Script to classify sentences as advice or non-advice'
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('--batch', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epoch', type=int, help='Epoch to test on')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--model', type=str, default="bertweet", help='Which pretrained LM to use as default')
    parser.add_argument('--dev', action='store_true', help='test on dev set')
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
    
    # Load the AutoTokenizer with a normalization mode if the input Tweet is raw
    tokenizer = AutoTokenizer.from_pretrained("multi_finetuned-model_" + args.model + "_seed_" + str(args.seed) + "_epoch_" + str(args.epoch) + "/", use_fast=False, normalization=True)
    
    # Loading the data
    emot_ans = ['Admiration', 'Anger', 'Disgust', 'Fear', 'Interest', 'Joy', 'Sadness', 'Surprise']
    df = pd.read_csv('../data-annotation/maj_df_split.tsv', sep='\t')
    df['tweet_clean'] = df.apply(lambda x: replace_ent(tweet=x['tweet'], ent=x['mentname']), axis=1)
    df['group'] = df['group'].apply(lambda x: 0 if x==-1 else 1)
    df.replace({False: 0, True: 1}, inplace=True)

    # df = df.drop(df[df['Split'] == 'train'].sample(frac=.5).index)
    data = Dataset.from_pandas(df)
    data = data.map(lambda x: tokenizer(x['tweet'], truncation=True, padding="max_length", max_length=128), batched=True)
    

    if args.dev:
        test_df = df[df['Split']=='dev']
        test_data = data.filter(lambda x: x['Split']=='dev')
    else:
        test_df = df[df['Split']=='test']
        test_data = data.filter(lambda x: x['Split']=='test')
    test_data.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'group'] + emot_ans)

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch)
    
    #Define model
    model = MultiTaskModel().to(device)
    model.load_state_dict(torch.load("multi_finetuned-model_" + args.model + "_seed_" + str(args.seed) + "_epoch_" + str(args.epoch) + "/model.bin"))
    

    model.zero_grad()
    model.eval()
    
    sigm = torch.nn.Sigmoid()
    
    # Tracking variables 
    pred_labels_group = np.array([])
    true_labels_group = np.array([])
    pred_labels_emot = None
    true_labels_emot = None

    with torch.no_grad():
        for _, input_dict in enumerate(test_dataloader):
            b_input_ids = input_dict['input_ids'].to(device)
            b_input_mask = input_dict['attention_mask'].to(device)
            b_group = input_dict['group'].to(device)
            b_labels = torch.cat([input_dict[x].unsqueeze(1) for x in emot_ans], axis=1).to(torch.float32).to(device)
            
            for task in ['classification', 'labelling']:
                out = model(input_ids=b_input_ids, attention_mask=b_input_mask, task=task)
                if task=='labelling':
                    out_sigmoids = sigm(out).detach().cpu()
                    pred_emots = (out_sigmoids>0.5).int().numpy()
                    labels_flat = b_labels.detach().cpu().numpy()
                    if pred_labels_emot is None:
                        pred_labels_emot = pred_emots
                        true_labels_emot = labels_flat
                    else:
                        pred_labels_emot = np.append(pred_labels_emot, pred_emots, axis=0)
                        true_labels_emot = np.append(true_labels_emot, labels_flat, axis=0)
                else:
                    groups_flat = b_group.detach().cpu().numpy()
                    out_sftmax = torch.softmax(out, axis=1)
                    preds_group = torch.argmax(out_sftmax, axis=1).detach().cpu().numpy()

                    pred_labels_group = np.append(pred_labels_group, preds_group)
                    true_labels_group = np.append(true_labels_group,groups_flat)
                    
                    
    test_df['preds'] = pred_labels_group
    fscore_emot = f1_score(true_labels_emot, pred_labels_emot, average='micro', zero_division=0)
    fscore_group = f1_score(true_labels_group, pred_labels_group, average='micro', zero_division=0)
    print('\nGroup F1 Score:', np.round(fscore_group,3))
    
    adm_f1=f1_score(test_df[test_df['Admiration']==True]['group'], test_df[test_df['Admiration']==True]['preds'])
    print('F1 score on admiration tweets', np.round(adm_f1, 3))
    
    ang_f1=f1_score(test_df[test_df['Anger']==True]['group'], test_df[test_df['Anger']==True]['preds'])
    print('F1 score on anger tweets', np.round(ang_f1, 3))

    dis_f1=f1_score(test_df[test_df['Disgust']==True]['group'], test_df[test_df['Disgust']==True]['preds'])
    print('F1 score on disgust tweets', np.round(dis_f1, 3))
    
    print('\nEmo F1 Score:', np.round(fscore_emot,3))
    indiv_f1s = f1_score(true_labels_emot, pred_labels_emot, average=None, zero_division=0)
    print({emot_ans[i]: np.round(indiv_f1s[i], 3) for i in range(len(emot_ans))})
    
    true_zero = [np.all(x==0) for x in true_labels_emot]
    pred_zero = [np.all(x==0) for x in pred_labels_emot]
    null_f1 = f1_score(true_zero, pred_zero)
    print("\nNull emotions F1 score:", np.round(null_f1, 3))
    
    in_inds = np.where(true_labels_group==1)[0]
    in_true = true_labels_emot[in_inds]
    in_preds = pred_labels_emot[in_inds]
    in_f1s = f1_score(in_true, in_preds, average=None, zero_division=0)
    in_f1 = f1_score(in_true, in_preds, average='micro', zero_division=0)
    print("In-group F1-scores:", np.round(in_f1,3), {emot_ans[i]: np.round(in_f1s[i], 3) for i in range(len(emot_ans))})
    
    out_inds = np.where(true_labels_group==0)[0]
    out_true = true_labels_emot[out_inds]
    out_preds = pred_labels_emot[out_inds]
    out_f1s = f1_score(out_true, out_preds, average=None, zero_division=0)
    out_f1 = f1_score(out_true, out_preds, average='micro', zero_division=0)
    print("Out-group F1-scores:", np.round(out_f1,3), {emot_ans[i]: np.round(out_f1s[i], 3) for i in range(len(emot_ans))})
    print("===================================================")
        

        
        