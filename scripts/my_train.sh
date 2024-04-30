#!/bin/bash
# run graphmil models
conda activate clam_latest
python3 my_main.py --path_to_splits /home/jonathan/code/CLAM/Rebuttal_MEDIA/splits_MEN/ --path_to_labels /home/jonathan/code/CLAM/Rebuttal_MEDIA/labels_MEN.csv --path_to_graphs Rebuttal/graphs_MEN_ImageNet/simclr_files/ --checkpoint_path Rebuttal/ --exp_code GraphFormer_MEN_ImageNet --k_start 5 --k 10 --logging
python3 my_main.py --path_to_splits /home/jonathan/code/CLAM/Rebuttal_MEDIA/splits_MEN/ --path_to_labels /home/jonathan/code/CLAM/Rebuttal_MEDIA/labels_MEN.csv --path_to_graphs Rebuttal/graphs_MEN_UNI/simclr_files --checkpoint_path Rebuttal/ --exp_code GraphFormer_MEN_UNI --k 10 --logging
python3 my_main.py --path_to_splits /home/jonathan/code/CLAM/Rebuttal_MEDIA/splits_MEN/ --path_to_labels /home/jonathan/code/CLAM/Rebuttal_MEDIA/labels_MEN.csv --path_to_graphs Rebuttal/graphs_MEN_CONCH/simclr_files --checkpoint_path Rebuttal/ --exp_code GraphFormer_MEN_CONCH --k 10 --logging --embed_dim 512
python3 my_main.py --path_to_splits /home/jonathan/code/CLAM/Rebuttal_MEDIA/splits_MEN/ --path_to_labels /home/jonathan/code/CLAM/Rebuttal_MEDIA/labels_MEN.csv --path_to_graphs Rebuttal/graphs_MEN_TransPath/simclr_files --checkpoint_path Rebuttal/ --exp_code GraphFormer_MEN_TransPath --k 10 --logging --embed_dim 768
# run TransMIL
conda deactivate
conda activate transmil
python3 /home/jonathan/code/TransMIL/crossval.py --stage train --config /home/jonathan/code/TransMIL/MEN/MEN_CONCH.yaml
python3 /home/jonathan/code/TransMIL/crossval.py --stage train --config /home/jonathan/code/TransMIL/MEN/MEN_ImageNet.yaml
python3 /home/jonathan/code/TransMIL/crossval.py --stage train --config /home/jonathan/code/TransMIL/MEN/MEN_transpath.yaml
python3 /home/jonathan/code/TransMIL/crossval.py --stage train --config /home/jonathan/code/TransMIL/MEN/MEN_UNI.yaml
conda deactivate