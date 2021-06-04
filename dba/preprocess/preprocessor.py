"""
!mkdir /root/.kaggle
!echo '{"username":"vaibhavpulastya","key":"8287078bae0808e36d8481548d2320b6"}' > /root/.kaggle/kaggle.json
!chmod 600 /root/.kaggle/kaggle.json
!kaggle datasets download -d benhamner/nips-papers
from zipfile import ZipFile

def extract():
    with ZipFile('nips-papers.zip', 'r') as zip:
        zip.extractall()
"""
from random import shuffle
from transformers import AlbertTokenizer
# import tensorflow as tf
import pandas as pd
import nltk
from itertools import permutations 
import numpy as np
import random
def download_nltk():
    nltk.download('punkt')

max_sent_num=5

def preprocess(args):
    
    datapoints =read_text(args)
    
    
    numAbs = len(datapoints)
    #Train
    sentence_train,sentence_y_train,sent_num_list_train=reorder_and_split(datapoints,0,int((0.8*numAbs)), args)
    #Validation
    sentence_valid,sentence_y_valid,sent_num_list_valid=reorder_and_split(datapoints,int((0.8*numAbs)) , int((0.9*numAbs)), args)
    #Test
    sentence_test,sentence_y_test,sent_num_list_test=reorder_and_split(datapoints,int((0.9*numAbs))  , int( numAbs ), args)
 
    if args.example_type=='all_sent':
        num_sent=max_sent_num
    else:
        num_sent=args.d_mlp 
    train_tokenized_ids,train_tokenized_masks,valid_tokenized_ids,valid_tokenized_masks,test_tokenized_ids,test_tokenized_masks=tokenizer_data(sentence_train, sentence_valid, sentence_test,num_sent)
  
    save_data(args,args.train,train_tokenized_ids,train_tokenized_masks,sentence_y_train,sent_num_list_train)
    save_data(args,args.valid,valid_tokenized_ids,valid_tokenized_masks,sentence_y_valid,sent_num_list_valid)
    save_data(args,args.test,test_tokenized_ids,test_tokenized_masks,sentence_y_test,sent_num_list_test)
    


def save_data(args,data_path_list, tokenized_ids, tokenized_masks,sentence_y,sent_num_list  ):
    data_parent_dir=args.data_dir
    id_path,mask_path,sent_num_path,label_path=data_path_list
    data_input_ids_dir=data_parent_dir+"/"+id_path
    data_attention_masks_dir=data_parent_dir+"/"+mask_path
    sent_num_dir=data_parent_dir+"/"+sent_num_path
    labels_dir=data_parent_dir+"/"+label_path
    np.save(data_input_ids_dir,tokenized_ids)
    np.save(data_attention_masks_dir,tokenized_masks)
    np.save(sent_num_dir,sent_num_list)
    np.save(labels_dir,sentence_y)

def tokenizer_data(sentence_train, sentence_valid, sentence_test,num_sent):
    train_tokenized_ids,train_tokenized_masks=gen_id_and_mask(sentence_train,num_sent)
    valid_tokenized_ids,valid_tokenized_masks=gen_id_and_mask(sentence_valid,num_sent)
    test_tokenized_ids,test_tokenized_masks=gen_id_and_mask(sentence_test,num_sent)
    return train_tokenized_ids,train_tokenized_masks,valid_tokenized_ids,valid_tokenized_masks,test_tokenized_ids,test_tokenized_masks

def gen_id_and_mask(sentence,num_sent):
    sent_data= []
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    for x in sentence :
        sent_data += list(x)
         
    inputs = tokenizer(list(sent_data), return_tensors="pt", padding=True) 
    tokenized_ids = inputs.input_ids
    tokenized_masks = inputs.attention_mask
    tokenized_ids = tokenized_ids.numpy().reshape((len(tokenized_ids)//num_sent, num_sent, tokenized_ids.shape[1]))
    tokenized_masks = tokenized_masks.numpy().reshape((len(tokenized_masks)//num_sent, num_sent, tokenized_masks.shape[1]))
    return tokenized_ids,tokenized_masks

def reorder_and_split(datapoints,start,end, args):
    if args.example_type !='all_sent':
        num_sent=args.d_mlp
        permutation_list=gen_permutation(num_sent)
    
        
    sentence  = []
    sentence_y  = []
    sent_num_list =[]
    
    for i,x in enumerate(datapoints[start : end]):
        # curSents = []
        # for v in range(num_sent):
        #     curSents.append(x[v])
        curSents = np.array(x)
        if args.example_type =='all_sent':
            num_sent=len(curSents)
            order = list(range(num_sent))
            shuffle(order)
            newL = curSents[order]
            padded=[ '<pad>' for x in range(num_sent, max_sent_num)]
            newL=np.append(newL,padded)
            sent_num_list.append(num_sent)
            sentence.append(newL)
            sentence_y.append(np.array(order))
        else:
            for pi in range(len(permutation_list)):
                order = list(permutation_list[pi])
                newL = curSents[order]
                 
                sentence.append(newL)
                sentence_y.append(pi)
    print(f'after reorder, gen {len(sentence)} examples')
    return sentence ,sentence_y,sent_num_list

def gen_permutation(num_sent):
     
    permutation_list = list(permutations([i for i in range(num_sent)]))
    return permutation_list


def read_text(args):
    d3 = pd.read_csv(args.coarse_data_dir)
    datapoints = []  
     
    num_sent = args.d_mlp
 
    for text in d3['abstract']:
        sentences = (nltk.sent_tokenize(text))
        if args.example_type=='all_sent':
            sentence_num=len(sentences)
            if sentence_num>1 and sentence_num<=max_sent_num:
                
                example=[sentences[x] for x in range(0, sentence_num)]
                
                datapoints.append(example)
                 
        else:
            idx = 0
            while(len(sentences) >= idx + num_sent):      
                #for single sentence
                datapoints.append([sentences[x] for x in range(idx, idx + num_sent)])
                if args.example_type=='overlap':
                    idx += 1
                else:
                    idx+=num_sent
    
    print(f'generate {len(datapoints)} examples in total')
    datapoints =shuffle_data(datapoints )
    return datapoints

def shuffle_data(datapoints ):
    datapoints = np.array(datapoints)
 
    idx = [i for i in range(len(datapoints))]
    random.shuffle(idx)
    datapoints = datapoints[idx]
 
    return datapoints 


def count_sentence_length():
    sent_length_array=np.zeros(20)
    total_paragraph_num=0 
    total_sentence_num=0
    file =  'data/real/preprocess/papers.csv'   
    d3 = pd.read_csv(file)
    for text in d3['abstract']:
        sentences = (nltk.sent_tokenize(text))
        sent_length=len(sentences)
        if sent_length>1:
            total_paragraph_num+=1
            sent_length_array[sent_length]+=1
            total_sentence_num+=sent_length
    for sent_length,num in enumerate(sent_length_array):
        if(num>0):
            print(f'there are {num} paragraph with {sent_length} sentences')
    print(f'there are total {total_sentence_num} sentences in {total_paragraph_num} paragraph')
         

if __name__ == '__main__':
    count_sentence_length()

