'''
Multitask learning of inout group interpersonal label and emotions
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
from train_emot_labeller import multilabel_f1
from accelerate import Accelerator
import ipdb
import argparse
import os

# Logging level for datasets
loggingd.disable_progress_bar()

 # Set logging to show only errors for transformers
loggingt.set_verbosity_error()

class MultiTaskModel(torch.nn.Module):
    def __init__(self, n_classes, lm_model, weights=None):
        super(MultiTaskModel, self).__init__()
        
        self.encoder = AutoModel.from_pretrained(lm_model)
        self.dropout = torch.nn.Dropout(0.1)
        self.n_classes = n_classes
        
        # emot labelling heads
        self.pooler_emot = torch.nn.Linear(768, n_classes)
        
        # inout classification head
        self.pooler_inout = torch.nn.Linear(768,2)
        
        # Cross entropy loss with emot logits
        self.loss_fn_emot = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
        
        # Loss fn
        self.loss_fn_inout = torch.nn.CrossEntropyLoss()
        
        # sigmoid function to label emotion as on
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask, task, labels=None):
        
        # Take output from <s> token and then dropout
        outputs = self.dropout(self.encoder(input_ids, attention_mask)['pooler_output'])
        
        # Do task specific heads
        if task=='emot':
            logits = self.pooler_emot(outputs)
            preds = (self.sigmoid(logits) > 0.5).int()
            loss = self.loss_fn_emot(logits, labels) if (labels is not None) else None
        else:
            logits = self.pooler_inout(outputs)
            preds = (torch.argmax(logits, axis=1)).int()
            loss = self.loss_fn_inout(logits, labels) if (labels is not None) else None
        
        return {
            'logits': logits,
            'loss': loss,
            'preds': preds
        }

if __name__== "__main__":
    # initialize argument parser
    description = 'Script to perform joint labelling of emotions and classification into in-group and out-group of text'
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
    df['emot'] = df.apply(lambda x: x[emot_ans].values.astype(float), axis=1)
    
    # Load the AutoTokenizer with a normalization mode if the input Tweet is raw
    tokenizer = AutoTokenizer.from_pretrained(args.model, normalization=True)

    # Preprocess and get data ready
    data = Dataset.from_pandas(df)
    data = data.map(lambda x: tokenizer(x['tweet'], truncation=True, padding="max_length", max_length=128, return_token_type_ids=False), batched=True)
    
    train_data = data.filter(lambda x: x['Split']=='train')
    train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'group', 'emot'])
    
    dev_data = data.filter(lambda x: x['Split']=='dev')
    dev_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'group', 'emot'])

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch, shuffle=True)
    dev_dataloader = torch.utils.data.DataLoader(dev_data, batch_size=args.batch, shuffle=True)
    
    ## Find weighting for the emotion classification loss
    support = np.sum(df.loc[:, emot_ans].values, axis=0)
    total = np.sum(support)
    not_support = np.array([(total-x) for x in support])
    weights = support/not_support
    
    #Define model
    model = MultiTaskModel(
        n_classes=len(emot_ans),
        lm_model=args.model,
        weights=torch.tensor(weights)
    )

    # Learning parameters:
    adam_epsilon = 1e-8
    num_warmup_steps = 1
    num_epochs = args.epochs
    
    # Each training data point is done twice since its multitask
    num_training_steps = num_epochs * len(train_dataloader) * 2
    
    # Optimize only the pooler and classifier weights
    optimizer = torch.optim.AdamW([{'params': [x for n,x in model.named_parameters() if 'pooler' not in n], 'lr': 2e-5}, {'params': [x for n,x in model.named_parameters() if 'pooler' in n], 'lr': 5e-3}], eps=adam_epsilon, lr=2e-5)
    
    # PyTorch scheduler
    scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    train_dataloader, dev_dataloader, model, optimizer, scheduler = accelerator.prepare(train_dataloader, dev_dataloader, model, optimizer, scheduler)
    
    for ep in range(0,args.epochs):
        print("\n<" + "="*22 + F" Epoch {ep} "+ "="*22 + ">")
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()
        for _, input_dict in enumerate(train_dataloader):
            for task in ['group', 'emot']:
                model.zero_grad()
                optimizer.zero_grad()
                
                train_dict = {
                        'input_ids': input_dict['input_ids'],
                        'attention_mask': input_dict['attention_mask'],
                        'labels': input_dict[task],
                        'task': task
                    }
                outputs = model(**train_dict)
                
                # Autograd loss
                accelerator.backward(outputs['loss'])
                
                # Clip the norm of the gradients to 1.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
        # Put the model into evaluation mode
        model.eval()
        model.zero_grad()
        
        # Tracking variables
        predictions_group = np.array([])
        references_group = np.array([])
        predictions_emot = None
        references_emot = None
        loss = 0
        
        for _, input_dict in enumerate(dev_dataloader):
            for task in ['group', 'emot']:
                with torch.no_grad():
                    test_dict = {
                        'input_ids': input_dict['input_ids'],
                        'attention_mask': input_dict['attention_mask'],
                        'labels': input_dict[task],
                        'task': task
                    }
                    outputs = model(**test_dict)
                
                loss += outputs['loss'].detach().cpu().numpy()
                
                if task=='emot':
                    if predictions_emot is None:
                        predictions_emot = outputs['preds'].detach().cpu().numpy()
                        references_emot = test_dict['labels'].int().detach().cpu().numpy()
                    else:
                        predictions_emot = np.append(predictions_emot, outputs['preds'].detach().cpu().numpy(), axis=0)
                        references_emot = np.append(references_emot, test_dict['labels'].int().detach().cpu().numpy(), axis=0)
                else:
                    predictions_group = np.append(predictions_group, outputs['preds'].detach().cpu().numpy())
                    references_group = np.append(references_group, test_dict['labels'].detach().cpu().numpy())
                    
        f1_emot = multilabel_f1(references_emot, predictions_emot)
        f1_group = multilabel_f1(references_group, predictions_group)
        
        print("Dev Loss: ", np.round(loss/(len(dev_dataloader)*2), 3))
        print("Dev Group F1 Score:", np.round(f1_group['micro_f1']*100, 1))
        print("Dev Emo F1 Score:", np.round(f1_emot['micro_f1']*100, 1))
        for i, emot in enumerate(emot_ans):
            print(emot + ":" + str(np.round(f1_emot['individual_f1s'][i]*100,1)), end=', ')
        
        print("\n")
        
        # Save model
        path = 'saved_models/multi_' + args.model.split('/')[-1] + '_seed_' + str(args.seed) + '_epoch_' + str(ep) + '/'
        os.makedirs(path, exist_ok = True)
        torch.save(model.state_dict(), path  + '/model.bin')
        tokenizer.save_pretrained(path)
        
