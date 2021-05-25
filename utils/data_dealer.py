from itertools import permutations 
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch 



def load_data(args):
    #DATA LOADING
    num_sent=args.d_mlp
    # permutation list
    permutation_list = list(permutations([i for i in range(num_sent)]))

    #loading in data
    id_path,mask_path,label_path=args.train
    train_data_input_ids = np.load(id_path)
    train_data_attention_masks = np.load(mask_path)
    train_labels = np.load(label_path)

    val_id_path,val_mask_path,val_label_path=args.valid
    validation_data_input_ids = np.load(val_id_path)
    validation_data_attention_masks = np.load(val_mask_path)
    validation_labels = np.load(val_label_path)

    #relabling
    train_new_labels = []
    for sample_label in train_labels:
        perm = permutation_list[sample_label]
        train_new_labels.append(perm)

    val_new_labels = []
    for sample_label in validation_labels:
        perm = permutation_list[sample_label]
        val_new_labels.append(perm)


    train_new_labels = (torch.from_numpy(np.array(train_new_labels))+1)
    val_new_labels =  (torch.from_numpy(np.array(val_new_labels))+1) 

    #reshaper
    train_data_input_ids = torch.from_numpy(train_data_input_ids)
    train_data_attention_masks = torch.from_numpy(train_data_attention_masks)
    validation_data_input_ids = torch.from_numpy(validation_data_input_ids)
    validation_data_attention_masks = torch.from_numpy(validation_data_attention_masks)


    #DataLoader
    my_dataset = TensorDataset(train_data_input_ids,train_data_attention_masks, train_new_labels) 
    train_dataloader = DataLoader(my_dataset, batch_size=8, shuffle=True) 
    
    #DataLoader
    val_dataset = TensorDataset(validation_data_input_ids,validation_data_attention_masks, val_new_labels) 
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True) 
    return train_dataloader,val_dataloader


def reshaper(to_reshape):

    num_sent = to_reshape.shape[1]
    ref_lis = [[] for i in range(num_sent)]
    
    for i in range(to_reshape.shape[0]):
        for j in range(num_sent):
            ref_lis[j].append(to_reshape[i][j])
            
    for i in range(num_sent):
        ref_lis[i] = torch.from_numpy(np.array(ref_lis[i]))
    
    return ref_lis