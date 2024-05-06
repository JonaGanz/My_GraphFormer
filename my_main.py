#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd
import argparse

from utils.dataset import MyGraphDataset
from utils.lr_scheduler import LR_Scheduler
from tensorboardX import SummaryWriter
from helper import Trainer, Evaluator, collate
from option import Options

# from utils.saliency_maps import *

from models.GraphTransformer import Classifier
from models.weight_init import weight_init

from pathlib import Path
from torch.utils.data import DataLoader
import yaml

def write_args_to_yaml(args:dict,output_dir:str,filename:str):
    """
    Writes the contents of a dict to a YAML file.

    Args:
        args (dict): Dict with parsed arguments.
        output_dir (str): The directory where the YAML file will be saved.
        filename (str): The desired name for the output YAML file.
    """
    # check whether output_dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct the full path for the output YAML file
    output_path = os.path.join(output_dir, filename)
    
    # Write the dictionary to a YAML file
    with open(output_path, 'w') as yaml_file:
        yaml.dump(args, yaml_file)


def create_model(n_class:int, embed_dim:int = 512):
    return Classifier(n_class, GCN_input_dim=embed_dim)

def train_one_epoch(model, trainer, optimizer, train_dl, log_interval_local:int = 5):
    model.train()
    train_loss = 0.
    total = 0.
    for i_batch, sample_batched in enumerate(train_dl):
            #scheduler(optimizer, i_batch, epoch, best_pred)
            preds,labels,loss = trainer.train(sample_batched, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss
            total += len(labels)
            trainer.metrics.update(labels, preds)
            
            if (i_batch + 1) % log_interval_local == 0:
                print(f"[{i_batch + 1}/{len(train_dl)}] train loss: {train_loss / total:.3f}; agg acc: {trainer.get_scores():.3f}")
                # trainer.plot_cm()
    acc = trainer.get_scores()
    trainer.reset_metrics()
    return acc, train_loss / total
                
def val_one_epoch(model, evaluator, val_dl):
    with torch.no_grad():
        model.eval()
        total = 0.
        val_loss = 0.
        for sample_batched in val_dl:
            preds, labels, loss = evaluator.eval_test(sample_batched, model)
            
            total += len(labels)
            val_loss += loss
            evaluator.metrics.update(labels, preds)

        print(f'val agg loss: {val_loss/total:.3f}; val agg acc: {evaluator.get_scores():.3f}')
        # evaluator.plot_cm()
        
        acc = evaluator.get_scores()
        evaluator.reset_metrics()
        return acc, val_loss / total

def train(
    model,
    train_dl,
    val_dl,
    num_epochs,
    fold_number:int,
    n_class:int = 244,
    lr:float = 1e-4,
    batch_size: int = 8,
    test:bool = False,
    model_path: Path = None,
    logging:bool = False,
    embed_dim:int = 1024,
    patience:int = 5
):
    # create logger if logging is True and test is False
    if logging and not test:
        writer = SummaryWriter(log_dir = model_path / f"{str(fold_number)}")
    else:
        writer = None
    # create trainer and evaluator
    trainer = Trainer(n_class, embed_dim = embed_dim)
    evaluator = Evaluator(n_class, embed_dim = embed_dim)
    # create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 5e-4)
    # get number of training samples
    total_train_num = len(train_dl.dataset) * batch_size
    # train for n epochs
    best_loss = np.inf
    best_acc = 0.
    early_stopping_counter = 0
    # early stopping flag
    stop = False
    for epoch in range(num_epochs):
        train_acc, train_loss = train_one_epoch(
            model = model,
            trainer = trainer,
            optimizer = optimizer,
            train_dl = train_dl)
        
        val_acc, val_loss = val_one_epoch(
            model = model,
            evaluator = evaluator, 
            val_dl = val_dl)
        
        if val_acc > best_acc:
            best_acc = val_acc
        if val_loss < best_loss: 
            if not test:
                print(f"{val_loss:.3f} is less than {best_loss:.3f}, saving model...")
                torch.save(model.state_dict(), model_path / f"{str(fold_number)}.pth")
            best_loss = val_loss
            early_stopping_counter = 0
        else:
            # early stopping
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered!")
                stop = True
        
        log = ""
        log = log + 'epoch [{}/{}] ------ acc: train = {:.3f}, val = {:.3f}'.format(epoch+1, num_epochs, train_acc, val_acc) + "\n"

        log += "================================\n"
        print(log)
        
        # log to tensorboard
        if logging and not test:
            writer.add_scalar('train/acc', train_acc, epoch)
            writer.add_scalar('val/acc', val_acc, epoch)
            writer.add_text('log', log, epoch)
            
        # if early stopping is triggered, break
        if stop:
            # return after early stopping
            return best_acc, best_loss.cpu().numpy()

    # return after n epochs
    return best_acc, best_loss.cpu().numpy()


def parse_args():
    parser = argparse.ArgumentParser(description='Graph Transformer')
    parser.add_argument('--path_to_splits', type=str, help = 'path to splits for crossvalidation')
    parser.add_argument('--path_to_labels', type=str, help = 'path to labels')
    parser.add_argument('--path_to_graphs', type=str, help = 'path to graphs')
    parser.add_argument('--checkpoint_path', type=str, help = 'path to save model')
    parser.add_argument('--exp_code', type=str, help = 'experiment code')
    
    parser.add_argument('--k', type=int, default = 5, help = 'k fold cross validation')
    parser.add_argument('--k_start', type=int, default = None, help = 'k fold start')
    parser.add_argument('--k_end', type=int, default = None, help = 'k fold end')
    parser.add_argument('--batch_size', type=int, default = 4, help = 'batch size')
    parser.add_argument('--device', type=str, default = 'cuda', help = 'device')
    parser.add_argument('--num_epochs', type=int, default = 200, help = 'number of epochs')
    parser.add_argument('--lr', type=float, default = 1e-4, help = 'learning rate')
    parser.add_argument('--test', action='store_true', default=False, help='test only')
    parser.add_argument('--logging', action='store_true', default=False, help='log only')
    parser.add_argument('--embed_dim', type=int, default = 1024, help = 'embedding dimension')
    parser.add_argument('--num_workers', type=int, default = 4, help = 'number of workers')
    parser.add_argument('--patience', type=int, default = 5, help = 'early stopping patience')
    parser.add_argument('--sparse_adj_matrix', action='store_true', default=False, help='use sparse adj matrix')
    return parser.parse_args()


def main(args):
    
    # make checkpoint path if it does not exist
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.exists(Path(args.checkpoint_path)/args.exp_code):
        os.makedirs(Path(args.checkpoint_path)/args.exp_code)
    
    # get the number of classes
    labels = pd.read_csv(args.path_to_labels)
    n_class = len(labels['label'].unique())
    
    
    # read the respective split
    start = args.k_start if args.k_start else 0
    end = args.k_end if args.k_end else args.k
    # store val_acc and val_loss
    val_res = []    
    for i in range(start, end):
        print(f"Starting fold {i}, loading split_{i}\n")
        # get splits for cross validation
        split_df = pd.read_csv(Path(args.path_to_splits) / f'splits_{i}.csv')
        
        train_df = split_df.merge(labels, left_on='train', right_on='slide_id')[['train','label']]
        train_df.columns = ['slide_id','label']
        val_df = split_df.merge(labels, left_on='val', right_on='slide_id')[['val','label']]
        val_df.columns = ['slide_id','label']
        test_df = split_df.merge(labels, left_on='test', right_on='slide_id')[['test','label']]
        test_df.columns = ['slide_id','label']
        
        if args.test:
            train_df = train_df[:10]
            
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        print(f"Initializing datasets....\n")
        # construct dataset and dataloader
        train_ds = MyGraphDataset(args.path_to_graphs, train_df, num_classes=n_class, load_from_sparse_tensor=args.sparse_adj_matrix)
        val_ds = MyGraphDataset(args.path_to_graphs, val_df, num_classes=n_class, load_from_sparse_tensor=args.sparse_adj_matrix)
        test_ds = MyGraphDataset(args.path_to_graphs, test_df, num_classes=n_class, load_from_sparse_tensor=args.sparse_adj_matrix)
        
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers)
        # test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers)
        print(f"Datasets initialized.\n")
        
        del train_ds, val_ds, test_ds
        # initialize model
        model = create_model(n_class, embed_dim = args.embed_dim)
        model = model.to(args.device)
        
        # start training
        print(f"Training fold {i}...\n")
        val_acc, val_loss = train(
            model = model,
            train_dl = train_dl,
            val_dl = val_dl,
            fold_number = i,
            num_epochs = args.num_epochs,
            n_class = n_class,
            lr = args.lr,
            batch_size = args.batch_size,
            test = args.test,
            model_path = Path(args.checkpoint_path)/args.exp_code,
            logging = args.logging,
            embed_dim = args.embed_dim,
            patience = args.patience,
            )
        
        val_res.append([i, val_acc, val_loss])
    
    # write results to file
    pd.DataFrame(val_res, columns = ['fold','val_acc','val_loss']).to_csv(Path(args.checkpoint_path)/args.exp_code / 'val_results.csv', index = False)
    print(f"Training completed.\n")
    
if __name__ == '__main__':
    args = parse_args()
    write_args_to_yaml(args, Path(args.checkpoint_path)/args.exp_code, 'config.yaml')
    main(args)