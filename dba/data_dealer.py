from itertools import permutations 
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch 
from torchtext import data 
from utils.config import * 

def load_data(args):
    #loading in data
    data_parent_dir=args.data_dir
    id_path,mask_path,sent_num_path,label_path=args.train
    train_data_input_ids = np.load(data_parent_dir+"/"+id_path)
    train_data_attention_masks = np.load(data_parent_dir+"/"+mask_path)
    train_sent_num =np.load(data_parent_dir+"/"+sent_num_path)
    train_labels = np.load(data_parent_dir+"/"+label_path,allow_pickle=True)

    val_id_path,val_mask_path,val_sent_num_path,val_label_path=args.valid
    validation_data_input_ids = np.load(data_parent_dir+"/"+val_id_path)
    validation_data_attention_masks = np.load(data_parent_dir+"/"+val_mask_path)
    validation_sent_num =np.load(data_parent_dir+"/"+val_sent_num_path)
    validation_labels = np.load(data_parent_dir+"/"+val_label_path,allow_pickle=True)


    #DATA LOADING
    train_new_labels,val_new_labels=gen_label(args,train_labels,validation_labels)
    

    

    #reshaper
    train_data_input_ids = torch.from_numpy(train_data_input_ids)
    train_data_attention_masks = torch.from_numpy(train_data_attention_masks)
    validation_data_input_ids = torch.from_numpy(validation_data_input_ids)
    validation_data_attention_masks = torch.from_numpy(validation_data_attention_masks)
    train_sentence_num = (torch.from_numpy(np.array(train_sent_num)))
    val_sentence_num =  (torch.from_numpy(np.array(validation_sent_num)) )


    #DataLoader
 
    my_dataset = TensorDataset(train_data_input_ids,train_data_attention_masks,train_sentence_num, train_new_labels) 
    val_dataset = TensorDataset(validation_data_input_ids,validation_data_attention_masks,val_sentence_num, val_new_labels) 

    
    train_dataloader = DataLoader(my_dataset, batch_size=8, shuffle=True) 
    
    #DataLoader
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True) 
    return train_dataloader,val_dataloader

def gen_label(args,train_labels,validation_labels):
    train_new_labels=gen_new_label(args,train_labels)
    val_new_labels=gen_new_label(args,validation_labels)
    return train_new_labels,val_new_labels

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