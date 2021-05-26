import argparse
 
import torch
import numpy as np
from torchtext import data
import logging
import random 
  
import time

from trainer import train, test
from pathlib import Path
import json
import os
from utils.timer import *
from utils.constants import *
from utils.saver import * 

def parse_args():
    parser = argparse.ArgumentParser(description='amazing idea')
  
    # environment
    parser.add_argument('--gpu_list', type=str, default="3")   
    parser.add_argument('--env', type=str, default="server",
                        choices=['server','colab' ])   

    # dataset settings
    parser.add_argument('--mode', type=str, default='train',
                        choices=['example','preprocess','train', 'test','hyper_search'
                                 'distill'])  # distill : take a trained AR model and decode a training set
    parser.add_argument('--data_dir', type=str,default='data/real/nips/content/sent_4')
    parser.add_argument('--train', type=str, nargs='+',default=["nips_valid_tokenized_ids.npy","nips_valid_tokenized_masks.npy","nips_valid_y.npy"])
    parser.add_argument('--valid', type=str, nargs='+',default=["nips_valid_tokenized_ids.npy","nips_valid_tokenized_masks.npy","nips_valid_y.npy"])
    parser.add_argument('--test', type=str, nargs='+',default=["nips_valid_tokenized_ids.npy","nips_valid_tokenized_masks.npy","nips_valid_y.npy"])

    # parser.add_argument('--train', type=str, nargs='+',default=["nips_train_tokenized_ids.npy","nips_train_tokenized_masks.npy","nips_train_y.npy"])
    # parser.add_argument('--valid', type=str, nargs='+',default=["data/real/nips/content/sent_4/nips_valid_tokenized_ids.npy","data/real/nips/content/sent_4/nips_valid_tokenized_masks.npy","data/real/nips/content/sent_4/nips_valid_y.npy"])
    # parser.add_argument('--test', type=str, nargs='+',default=["data/real/nips/content/sent_4/nips_valid_tokenized_ids.npy","data/real/nips/content/sent_4/nips_valid_tokenized_masks.npy","data/real/nips/content/sent_4/nips_valid_y.npy"])
    parser.add_argument('--max_len', type=int, default=None, help='limit the train set sentences to this many tokens')
    
    # settings for model
    parser.add_argument('--d_mlp', type=int, default=4, help='dimention size for MLP') 
    parser.add_argument('--early_stop', type=int, default=10)

    # setting for train 
    parser.add_argument('--seed', type=int, default=1234, help='seed for randomness')
    parser.add_argument('--batch_size', type=int, default=16, help='# of tokens processed per batch')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate')
    parser.add_argument('--max_epochs', type=int, default=10, help='maximum steps you take to train a model')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--output_parent_path', type=str, default="./")


    # setting for inference
    parser.add_argument('--load_from', nargs='+', default=None, help='load from 1.modelname, 2.lastnumber, 3.number')

    return parser.parse_args()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def override(args, load_dict, except_name):
    for k in args.__dict__:
        if k not in except_name:
            args.__dict__[k] = load_dict[k]

 
def mkdir_for_output(args):
    main_path = Path(args.output_parent_path)
    model_path, log_path=gen_next_output_path(main_path)
    for path in [model_path, log_path]:
        path.mkdir(parents=True, exist_ok=True)

def print_args(args):
    args_str = json.dumps(args.__dict__, indent=4, sort_keys=True)
    print(args_str)

def load_check_point(args):
    if len(args.load_from) == 1:
        load_from = '{}.best.pt'.format(args.load_from[0])
        print('{} load the best checkpoint from {}'.format(curtime(), load_from))
        checkpoint = torch.load(load_from, map_location='cpu')
        return checkpoint
    else:
        raise RuntimeError('must load model')

def recover_args(args,checkpoint):
    # when translate load_dict update args except some
    print('update args from checkpoint')
    load_dict = checkpoint['args'].__dict__
    except_name = ['mode', 'load_from', 'test', 'writetrans', 'beam_size', 'batch_size']
    override(args, load_dict, tuple(except_name))
    args_str = json.dumps(args.__dict__, indent=4, sort_keys=True)
    print(args_str)

def run_model(args):
    if args.mode == 'train' or args.mode=='example':
        mkdir_for_output(args)

        # setup random seeds
        set_seeds(args.seed)
 
        print_args(args)

        print('{} Start training'.format(curtime()))
        train(args )
    else:
        checkpoint=load_check_point(args)

        recover_args(args,checkpoint)
        
        start = time.time()
        test(args,  checkpoint)
        print('{} inference done, time {} mins'.format(curtime(), (time.time() - start) / 60))



def update_mutable_args(args):
    if args.env=="colab":
        args.data_dir="../../data/sent_4"
        args.train=["nips_valid_tokenized_ids.npy","nips_valid_tokenized_masks.npy","nips_valid_y.npy"]
        args.valid=["nips_valid_tokenized_ids.npy","nips_valid_tokenized_masks.npy","nips_valid_y.npy"]
        args.test=["nips_valid_tokenized_ids.npy","nips_valid_tokenized_masks.npy","nips_valid_y.npy"]

 
 


if __name__ == '__main__':
    
    args = parse_args()
    update_mutable_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_list
    run_model(args)