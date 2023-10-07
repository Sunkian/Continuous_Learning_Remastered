import os
import time
import argparse, configparser
from utils.args_loader import get_args
from utils import metrics
import torch
import faiss
import numpy as np
from exp.exp_OWL import Exp_OWL


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = get_args()

    exp = Exp_OWL(args) # set experiments
    print('>>>>>>>start feature extraction on in-distribution data : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.in_dataset))
    # exp.id_feature_extract()

    print('>>>>>>>start feature extraction on new-coming data : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.out_datasets))
    exp.ns_feature_extract('SVHN')

    print('>>>>>>>start ood detection on new-coming data : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.out_datasets))
    unknown_idx, bool_ood, scores_conf, pred_scores, pred_labels = exp.ood_detection('SVHN', K=50)

    print(f'Total new samples: {len(bool_ood)} \nNumber of correctly detected ood samples: {len(unknown_idx)}')
    torch.cuda.empty_cache()
