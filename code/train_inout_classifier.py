'''
Script to train models that can text as in-group or out-group
'''
import torch
import pandas as pd
import random
import numpy as np
import logging
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
    set_seed,
    logging as loggingt
)
from datasets import Dataset, logging as loggingd
import evaluate
from accelerate import Accelerator
import ipdb
import argparse

# Disable progress bar
loggingd.disable_progress_bar()

 # Set logging to show only errors for transformers
loggingt.set_verbosity_error()

if __name__== "__main__":
    # initialize argument parser
    description = 'Script to train models that can text as in-group or out-group'
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
        help='Max number of epochs'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='vinai/bertweet-base',
        help='Which pretrained LM to use as default. Should be valid huggingface model name'
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

    # Set seed
    set_seed(args.seed)
    random.seed(args.seed)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    
    # Initialise model, then load weights
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    
    # Load the AutoTokenizer with normalization mode
    # Normalization masks usernames and URLs
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, normalization=True)

    df = pd.read_csv(args.data_path, sep='\t')
    data = Dataset.from_pandas(df)
    data = data.map(lambda x: tokenizer(x['tweet'], truncation=True, padding="max_length", max_length=128), batched=True)
    # Rename group column to labels for use with BERT model
    data = data.rename_columns({'group': 'labels'})

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
    lr = 2e-5
    num_training_steps = num_epochs * len(train_dataloader)
    
    # Optimize only the classifier weights
    optimizer = torch.optim.AdamW([{'params': [x for n,x in model.named_parameters() if 'classifier' not in n], 'lr':args.lr_tr}, {'params': [x for n,x in model.named_parameters() if 'classifier' in n], 'lr': 5e-3}], eps=adam_epsilon, lr=lr)
    
    # PyTorch scheduler
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    # Load the f1 metric
    f1_metric = evaluate.load('f1')
    
    train_dataloader, dev_dataloader, model, optimizer, scheduler = accelerator.prepare(train_dataloader, dev_dataloader, model, optimizer, scheduler)

    for ep in range(0,args.epochs):
        print("\n<" + "="*22 + F" Epoch {ep} "+ "="*22 + ">")
        
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()
        
        for _, input_dict in enumerate(train_dataloader):
            optimizer.zero_grad()
            model.zero_grad()
            
            outputs = model(**input_dict)
            
            loss = outputs['loss']
            accelerator.backward(loss)
            
            # Clip the norm of the gradients to 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
        # Put the model into evaluation mode
        model.eval()
        model.zero_grad()
        
        loss = 0
        for _, input_dict in enumerate(dev_dataloader):
            with torch.no_grad():
                outputs = model(**input_dict)
                
            predictions = outputs['logits'].argmax(dim=1)
            
            loss += outputs['loss'].detach().cpu().numpy()
            
            # Store predictions and references for evaluation later
            predictions, references = accelerator.gather_for_metrics((predictions, input_dict['labels']))
            f1_metric.add_batch(
                predictions=predictions,
                references=references,
            )
        print("Dev Loss: ", np.round(loss/len(dev_dataloader), 3))
        print("Dev F1 Score: ", np.round(f1_metric.compute(average='micro')['f1']*100, 1))
        
        
        # Save models
        if args.lr_tr != 0:
            path_append = ""
        else:
            path_append = "_noft"
        model.save_pretrained("saved_models/" + args.model_name.split('/')[-1] + "_seed_" + str(args.seed) + "_epoch_" + str(ep) + path_append +"/")
        tokenizer.save_pretrained("saved_models/" + args.model_name.split('/')[-1] + "_seed_" + str(args.seed) + "_epoch_" + str(ep) + path_append + "/")
