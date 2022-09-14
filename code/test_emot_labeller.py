'''
Script to label tweet with interpersonal emotions
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


class emo_labeller(torch.nn.Module):
    def __init__(self, n_classes, bert_model):
        super(emo_labeller, self).__init__()
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.1)
        self.pooler = torch.nn.Linear(768, n_classes)

    def forward(self, input_ids, attention_mask):
        output_1 = self.bert(input_ids, attention_mask)["pooler_output"]
        output_2 = self.dropout(output_1)
        output = self.pooler(output_2)
        return output

def replace_ent(tweet, ent):
    'Find entity name in tweet and replace with placeholder' 
    
    pattern = re.compile(r"\@" + ent, re.IGNORECASE)
    return re.sub(pattern, "@USER", tweet)

if __name__== "__main__":
    # initialize argument parser
    description = 'Script to classify sentences as advice or non-advice'
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('--batch', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epoch', type=int, help='Epoch to test on')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--model', type=str, default="bertweet", help='Which pretrained LM to use as default')
    parser.add_argument('--lr_tr', type=float, default=2e-5, help='Set learning rate for BERT parameters')
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

    emot_ans = ['Admiration', 'Anger', 'Disgust', 'Fear', 'Interest', 'Joy', 'Sadness', 'Surprise']

    # Initialise model, then load weights
    models_to_test = {'bert': 'bert-base-uncased', 
                      'bertweet': 'vinai/bertweet-base',
                      'electra': 'google/electra-base-discriminator'}
    model_name = models_to_test[args.model]
    base_model = AutoModel.from_pretrained(model_name).to(device)
    model = emo_labeller(bert_model=base_model, n_classes=len(emot_ans)).to(device)
    
    if args.lr_tr != 0:
        path_append = ""
    else:
        path_append = "_noft"
        
    model.load_state_dict(torch.load("emot_finetuned-model_" + args.model + "_seed_" + str(args.seed) + "_epoch_" + str(args.epoch) + path_append + "/model.bin"))
    
    # Load the AutoTokenizer with a normalization mode if the input Tweet is raw
    tokenizer = AutoTokenizer.from_pretrained("emot_finetuned-model_" + args.model + "_seed_" + str(args.seed) + "_epoch_" + str(args.epoch) + path_append + "/", use_fast=False, normalization=True)
    
    # Loading the data
    df = pd.read_csv('../data-annotation/maj_df_split.tsv', sep='\t')
    df['tweet_clean'] = df.apply(lambda x: replace_ent(tweet=x['tweet'], ent=x['mentname']), axis=1)
    df['group'] = df['group'].apply(lambda x: 0 if x==-1 else 1)
    df.replace({False: 0, True: 1}, inplace=True)

    # df = df.drop(df[df['Split'] == 'train'].sample(frac=.5).index)
    data = Dataset.from_pandas(df)
    data = data.map(lambda x: tokenizer(x['tweet'], truncation=True, padding="max_length", max_length=128), batched=True)
    
    if args.dev:
        test_data = data.filter(lambda x: x['Split']=='dev')
    else: 
        test_data = data.filter(lambda x: x['Split']=='test')
    test_data.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'group'] + emot_ans)

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch, shuffle=True)
    

    model.zero_grad()
    model.eval()
    
    sigm = torch.nn.Sigmoid()
 
    # Tracking variables 
    pred_labels = None
    true_labels = None
    group_labels = None
    with torch.no_grad():
        for _, input_dict in enumerate(test_dataloader):
            b_input_ids = input_dict['input_ids'].to(device)
            b_input_mask = input_dict['attention_mask'].to(device)
            b_group = input_dict['group'].detach().cpu().numpy()
            out = model(input_ids=b_input_ids, attention_mask=b_input_mask)
            
            out_sigmoids = sigm(out).detach().cpu()
            pred_flat = (out_sigmoids>0.5).int().numpy()
            labels_flat = torch.cat([input_dict[x].unsqueeze(1) for x in emot_ans], axis=1).numpy()
            
            if pred_labels is None:
                pred_labels = pred_flat
                true_labels = labels_flat
                group_labels = b_group
            else:
                pred_labels = np.append(pred_labels, pred_flat, axis=0)
                true_labels = np.append(true_labels, labels_flat, axis=0)
                group_labels = np.append(group_labels, b_group, axis=0)
    
    fscore = f1_score(true_labels, pred_labels, average='micro', zero_division=0)
    print('\nValidation F1 Score:', np.round(fscore,3))
    indiv_f1s = f1_score(true_labels, pred_labels, average=None, zero_division=0)
    print({emot_ans[i]: np.round(indiv_f1s[i], 3) for i in range(len(emot_ans))})
    
    true_zero = [np.all(x==0) for x in true_labels]
    pred_zero = [np.all(x==0) for x in pred_labels]
    null_f1 = f1_score(true_zero, pred_zero)
    print("\nNull emotions F1 score:", np.round(null_f1, 3))
    
    in_inds = np.where(group_labels==1)[0]
    in_true = true_labels[in_inds]
    in_preds = pred_labels[in_inds]
    in_f1s = f1_score(in_true, in_preds, average=None, zero_division=0)
    in_f1 = f1_score(in_true, in_preds, average='micro', zero_division=0)
    print("In-group F1-scores:", np.round(in_f1,3), {emot_ans[i]: np.round(in_f1s[i], 3) for i in range(len(emot_ans))})
    
    out_inds = np.where(group_labels==0)[0]
    out_true = true_labels[out_inds]
    out_preds = pred_labels[out_inds]
    out_f1s = f1_score(out_true, out_preds, average=None, zero_division=0)
    out_f1 = f1_score(out_true, out_preds, average='micro', zero_division=0)
    print("Out-group F1-scores:", np.round(out_f1,3), {emot_ans[i]: np.round(out_f1s[i], 3) for i in range(len(emot_ans))})
    print("===================================================")