from utils.enums import BertType
from utils.bert_util import gen_bert_model
from model.layers.differential_ranking import gen_rank_func
import torch.nn as nn
import numpy as np
import torch 
from utils.mask import gen_sentence_mask
from utils.enums import * 
import torch.nn.functional as F
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
 
        # self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_bert, nhead=8)
        # self.self_attn=torch.nn.MultiheadAttention(embed_dim =d_bert, num_heads =8  )
        # encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_bert, nhead=8)
        # self.encoder= torch.nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.transformer=torch.nn.Transformer(d_model=d_bert)
    def forward(self,out,sentence_num_list):
         
        
        
        src_key_padding_mask=gen_src_key_padding_mask(  out,sentence_num_list)
        # out,weight = self.self_attn(out, out, out,  
        #                       key_padding_mask=src_key_padding_mask)
        # out = self.encoder(out, 
        #                       src_key_padding_mask =src_key_padding_mask) 
        # out = self.encoder_layer(out, 
        #                       src_key_padding_mask =src_key_padding_mask) 
        out=self.transformer(out,out,src_key_padding_mask =src_key_padding_mask,
        tgt_key_padding_mask =src_key_padding_mask)
        out= torch.transpose(out, 0, 1)

 
        out = self.score_layer(out)
        out=torch.squeeze(out,dim=-1)
        if len(sentence_num_list)>1:
            out=mask(out,sentence_num_list)
        out = self.rank(out.cpu(), regularization_strength=1.0) 
 
        return out,None




class PointerNetwork(torch.nn.Module):
    def __init__(self  ):

        super(PointerNetwork, self).__init__()
        d_bert=768
        self.rank = gen_rank_func()
        self.score_layer = torch.nn.Linear(d_bert , 1)#

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_bert, nhead=8)
        self.encoder= torch.nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.linears = nn.ModuleList([nn.Linear( d_bert, d_bert),
                                      nn.Linear( d_bert * 2, d_bert),
                                      nn.Linear(d_bert, 1)])
        self.decoder = nn.LSTM(d_bert, d_bert, batch_first=True)

    def decode(self,target, tgt_len  ,sentence_local_h, sentences_h,context_h):
        context_h = context_h.unsqueeze(0)
        cn = torch.zeros_like(context_h)
        context_hcn = (context_h, cn)

        dec_inputs=gen_decode_input(sentence_local_h,target)
        packed_dec_inputs,sorted_hn,sorted_cn,ix=sort_and_pack(dec_inputs,tgt_len,context_hcn)
        
        packed_dec_outputs, _ = self.decoder(packed_dec_inputs, (sorted_hn, sorted_cn))

        dec_outputs=un_sort_and_pack(packed_dec_outputs,ix)

        
        e=self.gen_attention_score(sentence_local_h, sentences_h,dec_outputs)
        return e
    def gen_attention_score(self,sentence_local_h, sentences_h,dec_outputs):
            keyinput = torch.cat((sentence_local_h, sentences_h), -1)
            sentence_key = self.linears[1](keyinput)
            # B qN 1 H
            query = self.linears[0](dec_outputs).unsqueeze(2)
            # B 1 kN H
            sentence_key = sentence_key.unsqueeze(1)
            # B qN kN H
            e = torch.tanh(query + sentence_key)
            # B qN kN
            e = self.linears[2](e).squeeze(-1)
            return e

    def encode(self,sentences_local_h,sentence_num_list ):
        src_key_padding_mask=gen_src_key_padding_mask(  sentences_local_h,sentence_num_list)
        sentences_h = self.encoder(sentences_local_h, 
                              src_key_padding_mask =src_key_padding_mask) 
        sentences_h= torch.transpose(sentences_h, 0, 1)
        sentences_local_h= torch.transpose(sentences_local_h, 0, 1)
        context_h=self.gen_context_h(sentences_h,sentence_num_list)
        return sentences_local_h,sentences_h,context_h

    def forward(self,sentences_local_h,sentence_num_list,target):
        sentences_local_h,sentences_h,context_h=self.encode(sentences_local_h,sentence_num_list )
 
        # (batch_size,sent_num), (batch_size), (batch_size,sent_num,d_h),(batch_size,sent_num,d_h),(batch_size,d_h)
        e=self.decode(target, sentence_num_list,sentences_local_h, sentences_h,context_h)


        pointed_mask,target_mask,pointed_mask_by_target=gen_mask(e,target, sentence_num_list )
        e.masked_fill_(pointed_mask == 1, -1e9)
        e.masked_fill_(pointed_mask_by_target == 0, -1e9)
        m = nn.Softmax(dim=-1)
        output = m(e)

        logp = F.log_softmax(e, dim=-1)
        logp = logp.view(-1, logp.size(-1))
        criterion = nn.NLLLoss(reduction='none')
        loss = criterion(logp, target.contiguous().view(-1))
        target_mask = target_mask.view(-1)
        loss.masked_fill_(target_mask == 0, 0)
        loss = loss.sum()/target.size(0)
        # sentences_h = self.score_layer(sentences_h)
        # sentences_h=torch.squeeze(sentences_h,dim=-1)
        # if len(sentence_num_list)>1:
        #     sentences_h=mask(sentences_h,sentence_num_list)
        # sentences_h = self.rank(sentences_h.cpu(), regularization_strength=1.0) 
        #(batch_size,sent_num)
        return output,loss  #TODO 

    def gen_context_h(self,sentences_h,sentence_num_list):
        sentence_mean = sentences_h.sum(1) /sentence_num_list.view(sentence_num_list.shape[0],1)
        return sentence_mean



def gen_decode_input(sentence_local_h,target):
    start = sentence_local_h.new_zeros(sentence_local_h.size(0), 1, sentence_local_h.size(2))
    # B N-1 H
    dec_inputs = sentence_local_h[torch.arange(sentence_local_h.size(0)).unsqueeze(1), target[:, :-1].type(torch.LongTensor)]
    # B N H
    dec_inputs = torch.cat((start, dec_inputs), 1)
    return dec_inputs

def sort_and_pack(dec_inputs,tgt_len,context_hcn):
    pack_len=torch.clone(tgt_len)
    sorted_len, ix = torch.sort(pack_len.to("cpu"), descending=True)
    sorted_len[0]=dec_inputs.shape[1]
    
    sorted_dec_inputs = dec_inputs[ix]
    packed_dec_inputs = nn.utils.rnn.pack_padded_sequence(sorted_dec_inputs, sorted_len, True) 
    hn, cn = context_hcn
    sorted_hn = hn[:, ix]
    sorted_cn = cn[:, ix]
    return packed_dec_inputs,sorted_hn,sorted_cn,ix

def un_sort_and_pack(packed_dec_outputs,ix):
    dec_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_dec_outputs, True)
    _, recovered_ix = torch.sort(ix, descending=False)
    dec_outputs = dec_outputs[recovered_ix]
    return dec_outputs

def gen_mask(e,target,tgt_len):
    target=target.type(torch.LongTensor)
    # mask already pointed nodes
    pointed_mask = [e.new_zeros(e.size(0), 1, e.size(2)).byte()]
    for t in range(1, e.size(1)):
        # B
        tar = target[:, t - 1]
        # B kN
        pm = pointed_mask[-1].clone().detach()
        pm[torch.arange(e.size(0)), :, tar] = 1
        pointed_mask.append(pm)
    # B qN kN
    pointed_mask = torch.cat(pointed_mask, 1)

    # mask for padded sentences
    pointed_mask_by_target = pointed_mask.new_zeros(pointed_mask.size(0), pointed_mask.size(2))
    target_mask = pointed_mask.new_zeros(pointed_mask.size(0), pointed_mask.size(1))
    for b in range(target_mask.size(0)):
        pointed_mask_by_target[b, :tgt_len[b]] = 1
        target_mask[b, :tgt_len[b]] = 1
    pointed_mask_by_target = pointed_mask_by_target.unsqueeze(1).expand_as(pointed_mask)
    return pointed_mask,target_mask,pointed_mask_by_target


class Network(torch.nn.Module):
    def __init__(self, device1,device2,parallel,bert_type,predict_type):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Network, self).__init__()

        # bert_embedder=
        self.bert_embedder =BertEmbedder(bert_type)
        if predict_type=="pointer_network":
            self.order_ranker= PointerNetwork()
        else:   
            self.order_ranker= OrderRanker()
        if parallel =='model':
            self.bert_embedder.to(device2)
            self.order_ranker.to(device1) 
        self.device1=device1
        self.device2=device2
        self.parallel=parallel

    def forward(self, input_ids, attn_mask,sentence_num_list,target):
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
        
            
        sentence_num_list=sentence_num_list.to(self.device1)
        out,loss = self.order_ranker(out,sentence_num_list,target)
  
        return out.to(self.device1),loss



def mask(out,sentence_num_list):
    sentence_mask=gen_sentence_mask(out,sentence_num_list)
    out.masked_fill_(sentence_mask==False ,np.inf)
    return out

 
    
def gen_src_key_padding_mask(  out,sentence_num_list):
    sentence_mask=torch.ones( out.shape[1],out.shape[0], dtype=torch.bool,device=out.device)
    for i in range(len(sentence_num_list)):
        sentence_mask[i,:sentence_num_list[i]]=False
    return sentence_mask


 