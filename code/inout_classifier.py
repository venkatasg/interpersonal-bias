'''
Script to classify in-group or out-group
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

def replace_ent(tweet, ent):
    'Find entity name in tweet and replace with placeholder'

    pattern = re.compile(r"\@" + ent, re.IGNORECASE)
    return re.sub(pattern, "@USER", tweet)

if __name__== "__main__":
    # initialize argument parser
    description = 'Script to classify sentences as advice or non-advice'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '--batch',
        type=int,
        default=64,
        help='Batch size for training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Max number of epochs'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed'
    )
    parser.add_argument(
        '--model',
        type=str,
        default="bertweet",
        help='Which pretrained LM to use as default'
    )
    parser.add_argument(
        '--lr_tr',
        type=float,
        default=2e-5,
        help='Set learning rate for BERT parameters'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to data'
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

    models_to_test = {'bert': 'bert-base-uncased',
                      'bertweet': 'vinai/bertweet-base',
                      'electra': 'google/electra-base-discriminator'}
    model_name = models_to_test[args.model]
    # Initialise model, then load weights
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    # Load the AutoTokenizer with a normalization mode if the input Tweet is raw
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, normalization=True)

    df = pd.read_csv(args.data_path, sep='\t')
    df['tweet_clean'] = df.apply(lambda x: replace_ent(tweet=x['tweet'], ent=x['mentname']), axis=1)
    df['group'] = df['group'].apply(lambda x: 0 if x==-1 else 1)

    # df = df.drop(df[df['Split'] == 'train'].sample(frac=.5).index)
    data = Dataset.from_pandas(df)
    data = data.map(lambda x: tokenizer(x['tweet'], truncation=True, padding="max_length", max_length=128), batched=True)

    train_data = data.filter(lambda x: x['Split']=='train')
    train_data.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'group'])

    dev_data = data.filter(lambda x: x['Split']=='dev')
    dev_data.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'group'])

    test_data = data.filter(lambda x: x['Split']=='test')
    test_data.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'group'])


    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch, shuffle=True)
    dev_dataloader = torch.utils.data.DataLoader(dev_data, batch_size=args.batch, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()

    # Learning parameters:
    adam_epsilon = 1e-8
    num_warmup_steps = 1
    num_epochs = args.epochs
    lr = 2e-5
    num_training_steps = num_epochs * len(train_dataloader)

    # Optimize only the pooler and classifier weights
    optimizer = torch.optim.AdamW([{'params': [x for n,x in model.named_parameters() if 'pooler' not in n], 'lr':args.lr_tr}, {'params': [x for n,x in model.named_parameters() if 'pooler' in n], 'lr': 5e-3}], eps=adam_epsilon, lr=lr)
    # PyTorch scheduler
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    model.zero_grad()
    model.train()

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
            b_labels = input_dict['group'].to(device)

            model.zero_grad()
            out = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)

            loss = out['loss']
            loss.backward()

            # Clip the norm of the gradients to 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            del out

        # Put the model into evaluation mode
        model.eval()
        model.zero_grad()

        # Tracking variables
        pred_labels = np.array([])
        true_labels = np.array([])
        loss = 0
        with torch.no_grad():
            for _, input_dict in enumerate(dev_dataloader):
                b_input_ids = input_dict['input_ids'].to(device)
                b_input_mask = input_dict['attention_mask'].to(device)
                b_labels = input_dict['group'].to(device)

                out = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)

                loss += out['loss'].detach().cpu().numpy()
                logits = out['logits'].detach().cpu().numpy()

                pred_flat = np.argmax(logits, axis=1).flatten()

                labels_flat= input_dict['group'].to('cpu').numpy().flatten()

                pred_labels = np.append(pred_labels, pred_flat)
                true_labels = np.append(true_labels, labels_flat)
        fscore = f1_score(true_labels, pred_labels, average='micro')
        dev_loss.append(loss/len(dev_dataloader))

        if args.lr_tr != 0:
            path_append = ""
        else:
            path_append = "_noft"

        model.save_pretrained("saved_models/finetuned-model_" + args.model + "_seed_" + str(args.seed) + "_epoch_" + str(ep) + path_append +"/")
        tokenizer.save_pretrained("saved_models/finetuned-model_" + args.model + "_seed_" + str(args.seed) + "_epoch_" + str(ep) + path_append + "/")
        # if (dev_loss[-1] - dev_loss[-2] < 0.00001):
        #     trigger_times = 0
        #     model.save_pretrained("saved_models/finetuned-model_" + args.model + "_seed_" + str(args.seed) + "_epoch_" + str(ep) + path_append +"/")
        #     tokenizer.save_pretrained("saved_models/finetuned-model_" + args.model + "_seed_" + str(args.seed) + "_epoch_" + str(ep) + path_append + "/")
        # else:
        #     trigger_times += 1
        #     if trigger_times >= patience:
        #         sys.exit()
        #     else:
        #         model.save_pretrained("saved_models/finetuned-model_" + args.model + "_seed_" + str(args.seed) + "_epoch_" + str(ep) + path_append + "/")
        #         tokenizer.save_pretrained("saved_models/finetuned-model_" + args.model + "_seed_" + str(args.seed) + "_epoch_" + str(ep) + path_append + "/")

        print('\nValidation F1 Score:', np.round(fscore,3))
        print("===================================================")
