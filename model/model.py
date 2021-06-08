from utils.enums import BertType
from utils.bert_util import gen_bert_model
from model.layers.differential_ranking import gen_rank_func

import numpy as np
import torch 
from utils.mask import gen_mask
from utils.enums import * 
def freeze_part_bert(bert_model,freeze_layer_num):
    count = 0
    for p in bert_model.named_parameters():
        
        if (count<=freeze_layer_num):
            p[1].requires_grad=False    
        else:
            break
        count=count+1
        print(p[0], p[1].requires_grad)

class BertEmbedder(torch.nn.Module):
    def __init__(self,bert_type):
        super(BertEmbedder, self).__init__()
        self.bert_model =gen_bert_model(bert_type)
        freeze_part_bert(self.bert_model,20)
        self.bert_type=bert_type

    def forward(self, input_ids, attn_mask):
        cat_list = []
        for i in range(input_ids.shape[1]):
            bert_out = self.bert_model(input_ids = input_ids[:,i,:].long(), attention_mask = attn_mask[:,i,:].float())
            if self.bert_type==BertType.albert:
                bert_out=bert_out.pooler_output
            else:
                bert_out=bert_out[0][:,0,:]
            cat_list.append(bert_out)
        out = torch.stack(cat_list, 0)
        return out

class OrderRanker(torch.nn.Module):
    def __init__(self  ):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(OrderRanker, self).__init__()

        
 
        d_bert=768
 
        self.score_layer = torch.nn.Linear(d_bert , 1)#
        

        self.rank = gen_rank_func()
 
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_bert, nhead=8)
        # self.self_attn=torch.nn.MultiheadAttention(embed_dim =d_bert, num_heads =1  )
 
    def forward(self,out,sentence_num_list):
         
        
        
        src_key_padding_mask=gen_src_key_padding_mask(  out,sentence_num_list)
        # out = self.self_attn(out, out, out,  
        #                       key_padding_mask=src_key_padding_mask)[0]
        out = self.encoder_layer(out, 
                              src_key_padding_mask =src_key_padding_mask) 
        out= torch.transpose(out, 0, 1)

 
        out = self.score_layer(out)
        out=torch.squeeze(out,dim=-1)
        if len(sentence_num_list)>1:
            out=mask(out,sentence_num_list)
        out = self.rank(out.cpu(), regularization_strength=1.0) 
 
        return out


class Network(torch.nn.Module):
    def __init__(self, device1,device2,parallel,bert_type,num_sent =4):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Network, self).__init__()

        # bert_embedder=
        self.bert_embedder =BertEmbedder(bert_type)
        self.order_ranker= OrderRanker()
        if parallel =='model':
            self.bert_embedder.to(device2)
            self.order_ranker.to(device1) 
        self.device1=device1
        self.device2=device2
        self.parallel=parallel

    def forward(self, input_ids, attn_mask,sentence_num_list):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        if self.parallel =='model':
            input_ids = input_ids.to(self.device2)
            attn_masks = attn_mask.to(self.device2)
            out=self.bert_embedder( input_ids,attn_masks)
            out=out.to(self.device1)
            sentence_num_list=sentence_num_list.to(self.device1)
        else:
            input_ids = input_ids.to(self.device1)
            attn_masks = attn_mask.to(self.device1)
            out=self.bert_embedder( input_ids,attn_masks)
        
            
        
        out = self.order_ranker(out,sentence_num_list)
  
        return out.to(self.device1)



def mask(out,sentence_num_list):
    sentence_mask=gen_mask(out,sentence_num_list)
    out.masked_fill_(sentence_mask==False ,np.inf)
    return out

 
    
def gen_src_key_padding_mask(  out,sentence_num_list):
    sentence_mask=torch.ones( out.shape[1],out.shape[0], dtype=torch.bool,device=out.device)
    for i in range(len(sentence_num_list)):
        sentence_mask[i,:sentence_num_list[i]]=False
    return sentence_mask


 