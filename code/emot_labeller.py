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
    parser.add_argument('--epochs', type=int, default=20, help='Max number of epochs')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--model', type=str, default="bertweet", help='Which pretrained LM to use as default')
    parser.add_argument('--lr_tr', type=float, default=2e-5, help='Set learning rate for BERT parameters')
    parser.add_argument('--lr_cl', type=float, default=5e-3, help='Set learning rate for BERT parameters')
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
    
    models_to_test = {'bert': 'bert-base-uncased', 
                      'bertweet': 'vinai/bertweet-base',
                      'electra': 'google/electra-base-discriminator'}
    model_name = models_to_test[args.model]
    # Initialise model, then load weights
    base_model = AutoModel.from_pretrained(model_name).to(device)
    model = emo_labeller(bert_model=base_model, n_classes=len(emot_ans)).to(device)
    
    # Load the AutoTokenizer with a normalization mode if the input Tweet is raw
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, normalization=True)
    
    # Loading the data
    df = pd.read_csv('../data-annotation/maj_df_split.tsv', sep='\t')
    df['tweet_clean'] = df.apply(lambda x: replace_ent(tweet=x['tweet'], ent=x['mentname']), axis=1)
    df['group'] = df['group'].apply(lambda x: 0 if x==-1 else 1)
    df.replace({False: 0, True: 1}, inplace=True)

    # df = df.drop(df[df['Split'] == 'train'].sample(frac=.5).index)
    data = Dataset.from_pandas(df)
    data = data.map(lambda x: tokenizer(x['tweet'], truncation=True, padding="max_length", max_length=128), batched=True)
    
    train_data = data.filter(lambda x: x['Split']=='train')
    train_data.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'group'] + emot_ans)
    
    dev_data = data.filter(lambda x: x['Split']=='dev')
    dev_data.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'group'] + emot_ans)
    
    test_data = data.filter(lambda x: x['Split']=='test')
    test_data.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'group'] + emot_ans)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch, shuffle=True)
    dev_dataloader = torch.utils.data.DataLoader(dev_data, batch_size=args.batch, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch, shuffle=True)
    
    # Learning parameters:
    adam_epsilon = 1e-8
    num_warmup_steps = 1
    num_epochs = args.epochs
    lr = 2e-5
    num_training_steps = num_epochs * len(train_dataloader)
    
    ## Find weighting for the actual classification loss
    support = np.sum(df.loc[:, emot_ans].values, axis=0)
    total = np.sum(support)
    not_support = np.array([(total-x) for x in support])
    weights = support/not_support
    
    # Cross entropy loss with logits
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights).to(device))

    # Optimize only the pooler and classifier weights
    optimizer = torch.optim.AdamW([{'params': [x for n,x in model.named_parameters() if 'pooler' not in n], 'lr':args.lr_tr}, {'params': [x for n,x in model.named_parameters() if 'pooler' in n], 'lr': args.lr_cl}], eps=adam_epsilon, lr=lr)
    # PyTorch scheduler
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    model.zero_grad()
    model.train()
    
    sigm = torch.nn.Sigmoid()
    dev_loss = [100]
    patience = 3
    trigger_times = 0
    
    for ep in range(0,args.epochs):
        print("\n<" + "="*22 + F" Epoch {ep} "+ "="*22 + ">")

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        for _, input_dict in enumerate(train_dataloader):

            b_input_ids = input_dict['input_ids'].to(device)
            b_input_mask = input_dict['attention_mask'].to(device)
            b_labels = torch.cat([input_dict[x].unsqueeze(1) for x in emot_ans], axis=1).to(torch.float32).to(device)

            model.zero_grad()
            out = model(input_ids=b_input_ids, attention_mask=b_input_mask)

            # Calculate loss
            loss = loss_fn(out, b_labels)
            loss.backward()

            # Clip the norm of the gradients to 1.0 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Put the model into evaluation mode
        model.eval()
        model.zero_grad()
    
        # Tracking variables 
        pred_labels = None
        true_labels = None
        group_labels = None
        loss = 0
        with torch.no_grad():
            for _, input_dict in enumerate(dev_dataloader):
                b_input_ids = input_dict['input_ids'].to(device)
                b_input_mask = input_dict['attention_mask'].to(device)
                b_group = input_dict['group'].detach().cpu().numpy()
                b_labels = torch.cat([input_dict[x].unsqueeze(1) for x in emot_ans], axis=1).to(torch.float32).to(device)
                
                out = model(input_ids=b_input_ids, attention_mask=b_input_mask)
                
                # Calculate loss
                loss += loss_fn(out, b_labels).detach().cpu().numpy()
                
                out_sigmoids = sigm(out).detach().cpu()
                pred_flat = (out_sigmoids>0.5).int().numpy()
                labels_flat = b_labels.detach().cpu().numpy()
                
                if pred_labels is None:
                    pred_labels = pred_flat
                    true_labels = labels_flat
                    group_labels = b_group
                else:
                    pred_labels = np.append(pred_labels, pred_flat, axis=0)
                    true_labels = np.append(true_labels, labels_flat, axis=0)
                    group_labels = np.append(group_labels, b_group, axis=0)
        fscore = f1_score(true_labels, pred_labels, average='micro', zero_division=0)
        dev_loss.append(loss/len(dev_dataloader))
        
        if args.lr_tr != 0:
            path_append = ""
        else:
            path_append = "_noft"

        if (dev_loss[-1] - dev_loss[-2] < 0.00001): 
            trigger_times = 0
            path = "emot_finetuned-model_" + args.model + "_seed_" + str(args.seed) + "_epoch_" + str(ep) + path_append + "/"
            os.makedirs(path, exist_ok = True)
            torch.save(model.state_dict(),path  + "/model.bin")
            tokenizer.save_pretrained(path)
        else:
            trigger_times += 1
            if trigger_times >= patience:
                sys.exit()
            else:
                path = "emot_finetuned-model_" + args.model + "_seed_" + str(args.seed) + "_epoch_" + str(ep) + path_append + "/"
                os.makedirs(path, exist_ok = True)
                torch.save(model.state_dict(),path  + "/model.bin")
                tokenizer.save_pretrained(path)
        
        print('\nValidation F1 Score:', np.round(fscore,3))
        indiv_f1s = f1_score(true_labels, pred_labels, average=None, zero_division=0)
        print({emot_ans[i]: np.round(indiv_f1s[i], 3) for i in range(len(emot_ans))})
        
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