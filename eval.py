import argparse
import os
import pickle
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.GraphTransformer import Classifier
from helper import Evaluator, collate
from utils.dataset import MyGraphDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('--checkpoint_path',type = str, help = 'Path to the model checkpoint')
    parser.add_argument('--path_to_graphs', type = str, default = None, help = 'Path to the folder containing the graphs')
    parser.add_argument('--path_to_labels', type = str, default = None, help = 'Path to the csv file containing the labels')
    parser.add_argument('--path_to_splits', type = str, default = None, help = 'Path to the folder containing the splits')
    parser.add_argument('--k', type=int, default = 5, help = 'k fold cross validation')
    parser.add_argument('--k_start', type=int, default = None, help = 'k fold start')
    parser.add_argument('--k_end', type=int, default = None, help = 'k fold end')
    parser.add_argument('--device', type=str, default = 'cuda', help = 'device')
    parser.add_argument('--num_workers', type=int, default = 4, help = 'number of workers')
    parser.add_argument('--evaluate_on', type=str, default = 'test', choices=['train','val','test'], help = 'Dataset to evaluate on')
    parser.add_argument('--batch_size', type=int, default = 8, help = 'Batch size')
    
    return parser.parse_args()

def read_yaml(file_path):
    
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    
    return data


def main(args):
    # make resutls dir under checkpoint_path if it does not exist
    if not os.path.exists(args.checkpoint_path + '/results'):
        os.mkdir(args.checkpoint_path + '/results')
    # load config file from checkpoint
    config = read_yaml(args.checkpoint_path + '/config.yaml')
    print(config)
    # get info from config if no argument is provided
    path_to_labels = config['path_to_labels'] if args.path_to_labels is None else args.path_to_labels
    path_to_splits = config['path_to_splits'] if args.path_to_splits is None else args.path_to_splits
    path_to_graphs = config['path_to_graphs'] if args.path_to_graphs is None else args.path_to_graphs
    # initilize dataset
    
    # read the respective split
    start = args.k_start if args.k_start is not None else 0
    end = args.k_end if args.k_end is not None else args.k
    
    for i in range(start,end):
        print(f"Loading split {i}...\n")
        # get the number of classes
        labels_df = pd.read_csv(path_to_labels)
        n_class = len(labels_df['label'].unique())
        # load respective split
        split_df = pd.read_csv(Path(path_to_splits) / f'splits_{i}.csv')
        df = split_df.merge(labels_df, left_on=args.evaluate_on, right_on='slide_id')[[args.evaluate_on,'label']]
        df.columns = ['slide_id','label']
        df.reset_index(drop=True, inplace=True)
        # initilize dataset and dataloader
        ds = MyGraphDataset(path_to_graphs, df, num_classes=n_class)
        dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate)
        del ds,df
        # initilize model
        model = Classifier(n_class, GCN_input_dim=config['embed_dim'], return_logits=True)
        model.load_state_dict(torch.load(Path(args.checkpoint_path) / f'{i}.pth'))
        model.to(args.device)
        model.eval()
        # initilize evaluator
        evaluator = Evaluator(n_class, return_logits=True,embed_dim=config['embed_dim'])
        resutls = {}
        with torch.no_grad():
            for batch in tqdm(dl, total = len(dl), desc = f'Processing split {i}'):
                _, labels, _, logits = evaluator.eval_test(batch, model)      

                for label, logit, slide_name in zip(labels, logits, batch['id']):
                    if label not in resutls:
                        resutls[slide_name] = {
                            'label': label.cpu(),
                            'preds': logit.cpu()
                        }

        # save results
        with open(args.checkpoint_path + f'/results/fold_{i}_{args.evaluate_on}_df.p', 'wb') as f:
            pickle.dump(resutls, f) 
                        
        
    print('\nDone!')
    
if __name__ == '__main__':
    args = parse_args()
    main(args)