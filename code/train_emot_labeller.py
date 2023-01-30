'''
Script to label tweet with interpersonal emotions
'''
import torch
import pandas as pd
import random
import numpy as np
import logging
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_scheduler,
    set_seed,
    logging as loggingt
)
from datasets import Dataset, logging as loggingd
from accelerate import Accelerator
from sklearn.metrics import f1_score
import ipdb
import argparse
import os

# Logging level for datasets
loggingd.disable_progress_bar()

 # Set logging to show only errors for transformers
loggingt.set_verbosity_error()

def multilabel_f1(predictions, references):
    macro_f1 = f1_score(predictions, references,average='macro', zero_division=0)
    micro_f1 = f1_score(predictions, references,average='micro', zero_division=0)
    individual_f1s = f1_score(predictions, references,average=None, zero_division=0)
    
    return {'micro_f1': micro_f1, 'individual_f1s': individual_f1s, 'macro_f1': macro_f1}


class emo_labeller(torch.nn.Module):
    def __init__(self, n_classes, lm_model, weights):
        super(emo_labeller, self).__init__()
        self.encoder = AutoModel.from_pretrained(lm_model)
        self.dropout = torch.nn.Dropout(0.1)
        self.pooler = torch.nn.Linear(768, n_classes)
        
        # Cross entropy loss with logits
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights))
        
        # sigmoid function to label emotion as on
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask, labels=None):
        output_1 = self.encoder(input_ids, attention_mask)["pooler_output"]
        output_2 = self.dropout(output_1)
        logits = self.pooler(output_2)
        preds = (self.sigmoid(logits) > 0.5).int()
        loss = self.loss_fn(logits, labels) if (labels is not None) else None
        
        return {
            'logits': logits,
            'loss': loss,
            'preds': preds
        }

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
        '--epochs',
        type=int,
        default=20,
        help='Max number of epochs to train'
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
        default='vinai/bertweet-base',
        help='Which pretrained LM to use in emot labeller'
    )
    parser.add_argument(
        '--lr_tr',
        type=float,
        default=2e-5,
        help='Set learning rate for LM parameters'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to data'
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
        lm_model=args.model,
        n_classes=len(emot_ans),
        weights=weights
    )
    
    # Load the AutoTokenizer with a normalization mode if the input Tweet is raw
    tokenizer = AutoTokenizer.from_pretrained(args.model, normalization=True)

    # Preprocess and get data ready
    data = Dataset.from_pandas(df)
    data = data.map(lambda x: tokenizer(x['tweet'], truncation=True, padding="max_length", max_length=128), batched=True)

    train_data = data.filter(lambda x: x['Split']=='train')
    train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    dev_data = data.filter(lambda x: x['Split']=='dev')
    dev_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch, shuffle=True)
    dev_dataloader = torch.utils.data.DataLoader(dev_data, batch_size=args.batch, shuffle=True)
    
    # Learning parameters:
    adam_epsilon = 1e-8
    num_warmup_steps = 1
    num_epochs = args.epochs
    default_lr = 2e-5
    num_training_steps = num_epochs * len(train_dataloader)
    
    # Optimize only the pooler and classifier weights
    optimizer = torch.optim.AdamW([{'params': [x for n,x in model.named_parameters() if 'pooler' not in n], 'lr':args.lr_tr}, {'params': [x for n,x in model.named_parameters() if 'pooler' in n], 'lr': 5e-3}], eps=adam_epsilon, lr=default_lr)
    
    # PyTorch scheduler
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    # Load the f1 metric
    # f1_metric = evaluate.load('f1')
    
    train_dataloader, dev_dataloader, model, optimizer, scheduler = accelerator.prepare(train_dataloader, dev_dataloader, model, optimizer, scheduler)

    for ep in range(0,args.epochs):
        print("\n<" + "="*22 + F" Epoch {ep} "+ "="*22 + ">")
        
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()
        
        for _, input_dict in enumerate(train_dataloader):
            optimizer.zero_grad()
            model.zero_grad()
            
            outputs = model(**input_dict)
            
            # Calculate loss
            accelerator.backward(outputs['loss'])
            
            # Clip the norm of the gradients to 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()

        # Put the model into evaluation mode
        model.eval()
        model.zero_grad()
    
        # Tracking loss
        loss = 0
        predictions = None
        references = None
        
        for _, input_dict in enumerate(dev_dataloader):
            with torch.no_grad():
                outputs = model(**input_dict)
                
            # Calculate loss
            loss += outputs['loss'].detach().cpu().numpy()
            
            if predictions is None:
                predictions = outputs['preds'].detach().cpu().numpy()
                references = input_dict['labels'].int().detach().cpu().numpy()
            else:
                predictions = np.append(predictions, outputs['preds'].detach().cpu().numpy(), axis=0)
                references = np.append(references, input_dict['labels'].int().detach().cpu().numpy(), axis=0)
        
        f1_metric = multilabel_f1(predictions, references)
        print("Dev Loss:", np.round(loss/len(dev_dataloader),3))
        print("Dev F1 Score:", np.round(f1_metric['micro_f1']*100, 1))
        for i, emot in enumerate(emot_ans):
            print(emot + ":" + str(np.round(f1_metric['individual_f1s'][i]*100,1)), end=', ')
        
        print("\n")
        
        # Save finetuned model
        if args.lr_tr != 0:
            path_append = ""
        else:
            path_append = "_noft"
        path = "saved_models/emot_" + args.model.split('/')[-1] + "_seed_" + str(args.seed) + "_epoch_" + str(ep) + path_append + "/"
        os.makedirs(path, exist_ok = True)
        torch.save(model.state_dict(), path  + "/model.bin")
        tokenizer.save_pretrained(path)
