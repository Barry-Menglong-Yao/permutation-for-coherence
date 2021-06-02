from model.layers.differential_ranking import gen_rank_func
from transformers import AlbertConfig,AlbertModel
import numpy as np
import torch 


def freeze_part_bert(bert_model,freeze_layer_num):
    count = 0
    for p in bert_model.named_parameters():
        
        if (count<=freeze_layer_num):
            p[1].requires_grad=False    
            print(p[0], p[1].requires_grad)
        else:
            break
        count=count+1

class Network(torch.nn.Module):
    def __init__(self, num_sent = 4):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Network, self).__init__()
        self.bert_model = AlbertModel.from_pretrained('albert-base-v2') 

        d_bert=768
        self.linear_1 = torch.nn.Linear(d_bert, d_bert)
        self.linear_2 = torch.nn.Linear(1024*num_sent,4096 )
        self.score_layer = torch.nn.Linear(d_bert , 1)#
        self.num_sent=num_sent

        self.rank = gen_rank_func()
        self.drop1 = torch.nn.Dropout(p=0.5)
        self.drop2 = torch.nn.Dropout(p=0.5 )
        freeze_part_bert(self.bert_model,20)
        self.transformer_encoder_layer=torch.nn.TransformerEncoderLayer(d_model=d_bert, nhead=num_sent )

    def forward(self, input_ids, attn_mask):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """


        cat_list = []
        for i in range(self.num_sent):
            bert_out = self.bert_model(input_ids = input_ids[:,i,:].long(), attention_mask = attn_mask[:,i,:].float()).pooler_output
            # shared_out = torch.relu(self.linear_1(bert_out))
            # shared_out=self.drop1(shared_out)
            # cat_list.append(shared_out)
            cat_list.append(bert_out)
        # out = torch.cat(cat_list, 1)
        out = torch.stack(cat_list, 0)
        out =  self.transformer_encoder_layer(out)
        # (num_sent,batch_size) -> (batch_size,num_sent)
        out= torch.transpose(out, 0, 1)
        # out = torch.relu(self.linear_2(out))
        # out = self.drop2(out)
 
        out = self.score_layer(out)
        out=torch.squeeze(out)
        out = self.rank(out.cpu(), regularization_strength=1.0) 

        return out