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
import dba.preprocess.preprocessor  as preprocessor
from utils.log import time_flag 
from utils.log import logger

def parse_args():
    parser = argparse.ArgumentParser(description='amazing idea')
  
    # environment
    parser.add_argument('--gpu', type=str, default="1")  
    parser.add_argument('--gpu2', type=str, default="3")    
    parser.add_argument('--env', type=str, default="server",
                        choices=['server','colab' ])   

    # dataset settings
    parser.add_argument('--mode', type=str, default='train',
                        choices=['example','preprocess','train', 'test','hyper_search'
                                 'distill'])  # distill : take a trained AR model and decode a training set
    parser.add_argument('--example_type', type=str, default="all_sent", 
                        choices=['overlap','part_sent','all_sent' ],help='overlap')
    parser.add_argument('--data_dir', type=str,default='data/real/nips/content/all_sent_5')
    parser.add_argument('--train', type=str, nargs='+',default=["nips_train_tokenized_ids.npy","nips_train_tokenized_masks.npy","nips_train_sent_num.npy","nips_train_y.npy"])
    parser.add_argument('--valid', type=str, nargs='+',default=["nips_valid_tokenized_ids.npy","nips_valid_tokenized_masks.npy","nips_valid_sent_num.npy","nips_valid_y.npy"])
    parser.add_argument('--test', type=str, nargs='+',default=["nips_test_tokenized_ids.npy","nips_test_tokenized_masks.npy","nips_test_sent_num.npy","nips_test_y.npy"])
    parser.add_argument('--max_len', type=int, default=None, help='limit the train set sentences to this many tokens')
    
    # settings for model
    parser.add_argument('--d_mlp', type=int, default=4, help='dimention size for MLP') 
    

    # setting for train 
    #input: data_dir
    #output: output_parent_dir (model, log)
    parser.add_argument('--output_parent_dir', type=str, default="./")
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1234, help='seed for randomness')
    parser.add_argument('--batch_size', type=int, default=16, help='# of tokens processed per batch')
    parser.add_argument('--lr', type=float, default=1e-6, help='learning rate')
    parser.add_argument('--max_epochs', type=int, default=200, help='maximum steps you take to train a model')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--amp', type=str, default="N", help='amp')
    parser.add_argument('--remark', type=str,  help='describe experiment setting')
    parser.add_argument('--metric', type=str, default='acc',
                        choices=['pmr','acc','taus' ])
    parser.add_argument('--parallel', type=str, default='none',
                        choices=['model','function','none' ])                    

    # setting for inference
    #input: load_from, data_dir
    parser.add_argument('--load_from', nargs='+', default=None, help='load from 1.modelname, 2.lastnumber, 3.number')
    #output: None

    # preprocess setting 
    #input: coarse_data
    parser.add_argument('--coarse_data_dir', type=str,default='data/real/preprocess/papers.csv')
    
    # parser.add_argument('--task', type=str, default="permutation", help='task',
    #                     choices=['permutation','sentence_order'  ])
    
    #output: data_dir

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
    main_path = Path(args.output_parent_dir)
    model_path, log_path=gen_next_output_path(main_path)
    for path in [model_path, log_path]:
        path.mkdir(parents=True, exist_ok=True)

def print_args(args):
    args_str = json.dumps(args.__dict__, indent=4, sort_keys=True)
    print(args_str)
    logger.info(f'data_dir:{args.data_dir}' )

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
    elif args.mode=="test":
        checkpoint=load_check_point(args)

        recover_args(args,checkpoint)
        
        start = time.time()
        test(args,  checkpoint)
        print('{} inference done, time {} mins'.format(curtime(), (time.time() - start) / 60))
    elif args.mode=="preprocess":
        data_path = Path(args.data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        preprocessor.preprocess(args )


def update_mutable_args(args):
    if args.env=="colab":
        args.data_dir="data/sent_4"
        args.coarse_data_dir="data/sent_4"
    args.__dict__["time_flag"]=time_flag

    if args.parallel =='none':
        args.__dict__["gpu2"]=args.gpu


if __name__ == '__main__':
    
    args = parse_args()
    update_mutable_args(args)
    gpu_list=args.gpu+","+args.gpu2
    if args.parallel =='none':
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    run_model(args)