from itertools import permutations 
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch 
from torchtext import data 
from utils.config import * 
import sklearn.model_selection as model_selection
from random import shuffle
import pandas as pd
import nltk
from torch.utils.data import random_split
from utils.bert_util import *
from dba.dataset import NipsDataset 

def load_data(args):
    if args.example_type=='all_sent':
        train_dataset,val_dataset=load_data_from_csv(args)
    else:
        #loading in data
        data_parent_dir=args.data_dir
        id_path,mask_path,sent_num_path,label_path=args.train
        train_data_input_ids = np.load(data_parent_dir+"/"+id_path)
        train_data_attention_masks = np.load(data_parent_dir+"/"+mask_path)
        # train_sent_num =np.load(data_parent_dir+"/"+sent_num_path)
        train_labels = np.load(data_parent_dir+"/"+label_path,allow_pickle=True)

        val_id_path,val_mask_path,val_sent_num_path,val_label_path=args.valid
        validation_data_input_ids = np.load(data_parent_dir+"/"+val_id_path)
        validation_data_attention_masks = np.load(data_parent_dir+"/"+val_mask_path)
        # validation_sent_num =np.load(data_parent_dir+"/"+val_sent_num_path)
        validation_labels = np.load(data_parent_dir+"/"+val_label_path,allow_pickle=True)


        #DATA LOADING
        
        train_new_labels,val_new_labels=gen_label(args,train_labels,validation_labels)
        train_sent_num=gen_sent_num(args,train_labels)
        validation_sent_num=gen_sent_num(args,validation_labels)

        

        #reshaper
        train_data_input_ids = torch.from_numpy(train_data_input_ids)
        train_data_attention_masks = torch.from_numpy(train_data_attention_masks)
        validation_data_input_ids = torch.from_numpy(validation_data_input_ids)
        validation_data_attention_masks = torch.from_numpy(validation_data_attention_masks)
        train_sentence_num = (torch.from_numpy(np.array(train_sent_num)))
        val_sentence_num =  (torch.from_numpy(np.array(validation_sent_num)) )


        #DataLoader
    
        train_dataset = TensorDataset(train_data_input_ids,train_data_attention_masks,train_sentence_num, train_new_labels) 
        val_dataset = TensorDataset(validation_data_input_ids,validation_data_attention_masks,val_sentence_num, val_new_labels) 

    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) 
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True) 
    return train_dataloader,val_dataloader


def read_text(args):
    datapoints = []  
    sentence_num_list=[]
    sentence_y  = []
    d3 = pd.read_csv(args.coarse_data_dir)
    tokenizer=gen_tokenizer(args.bert_type)
    for text in d3['abstract']:
        sentences = (nltk.sent_tokenize(text))
        sentence_num=len(sentences)
        if sentence_num>1 and sentence_num<args.max_sent_num:
            #  if(is_not_cut(tokenizer,args.max_len,sentences,args.bert_type)):
            shuffle_sents,order=shuffle_and_pad(sentences,sentence_num,args.max_sent_num )
            datapoints.append(shuffle_sents)
            sentence_y.append(order)
            sentence_num_list.append(sentence_num)
    return datapoints,sentence_num_list,sentence_y



def is_not_cut(tokenizer, max_sent_length,sentences,bert_type):
    
    inputs = tokenizer(sentences, return_tensors="pt", padding=True,return_length =True) 
    token_len_list=inputs.length 
    for token_len in token_len_list:
        if token_len>max_sent_length:
            return False
    return True

def load_data_from_csv(args):
    datapoints,sentence_num_list,sentence_y=read_text(args)
    max_len=args.max_len 
    tokenizer = gen_tokenizer(args.bert_type)
    total_dataset=NipsDataset(datapoints,sentence_num_list,sentence_y,tokenizer,max_len)

    total_len = len(datapoints)
    train_len = int(0.8*total_len)
    valid_len = int(0.1*total_len)
    train_dataset,val_dataset,test_dataset=random_split(total_dataset, [train_len, valid_len,total_len-train_len- valid_len] )
    return train_dataset,val_dataset

def shuffle_and_pad(sentences,sentence_num,max_sent_num):
    cur_sents = np.array(sentences)
    order = list(range(sentence_num))
    shuffle(order)
    shuffle_sents = cur_sents[order]

    padded=[ '<sent_pad>' for x in range(sentence_num, max_sent_num)]
    shuffle_sents=np.append(shuffle_sents,padded)
    padded_order=[ 0 for x in range(sentence_num, max_sent_num)]
    order.extend(padded_order)
    return shuffle_sents.tolist(),order


def gen_label(args,train_labels,validation_labels):
    train_new_labels=gen_new_label(args,train_labels)
    val_new_labels=gen_new_label(args,validation_labels)
    return train_new_labels,val_new_labels

def gen_sent_num(args, labels):
    sent_num_list =[]
    for label in  labels:
        if args.example_type=='all_sent':
            num_sent=len(label)
        else: 
            num_sent=args.d_mlp
        sent_num_list.append(num_sent)
    return sent_num_list

def gen_new_label(args,train_labels):
    
    if args.example_type=='all_sent':
        
        ORDER = data.Field(batch_first=True, include_lengths=True, pad_token=0, use_vocab=False,
                        sequential=True)
        train_new_labels,label_length=ORDER.process(train_labels, device=DEVICE)

    else:
        num_sent=args.d_mlp
        # permutation list
        permutation_list = list(permutations([i for i in range(num_sent)]))
        #relabling
        train_new_labels = []
        
        for sample_label in train_labels:
            perm = permutation_list[sample_label]
            train_new_labels.append(perm)

         
        train_new_labels = (torch.from_numpy(np.array(train_new_labels))+1)  
 
    return train_new_labels 

def reshaper(to_reshape):

    num_sent = to_reshape.shape[1]
    ref_lis = [[] for i in range(num_sent)]
    
    for i in range(to_reshape.shape[0]):
        for j in range(num_sent):
            ref_lis[j].append(to_reshape[i][j])
            
    for i in range(num_sent):
        ref_lis[i] = torch.from_numpy(np.array(ref_lis[i]))
    
    return ref_lis