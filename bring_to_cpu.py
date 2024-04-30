import torch 
import os
from tqdm import tqdm

path = 'Rebuttal/graphs_MEN_CONCH/simclr_files'

for dirpath, dirnames, filenames in tqdm(os.walk(path)):
    for filename in filenames:
        if 'adj_s' in filename:
            x = torch.load(os.path.join(dirpath, filename))
            x = x.cpu()
            torch.save(x, os.path.join(dirpath, filename))
            break
