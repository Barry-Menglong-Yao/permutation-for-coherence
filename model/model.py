from transformers import AlbertModel
import numpy as np
import torch
import torchsort

class Network(torch.nn.Module):
    def __init__(self, num_sent = 4):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Network, self).__init__()
        self.bert_model = AlbertModel.from_pretrained('albert-base-v2') 
        self.linear_1 = torch.nn.Linear(768, 768)
        self.linear_2 = torch.nn.Linear(768*num_sent, 1024)
        self.linear_3 = torch.nn.Linear(1024, 512)
        self.linear_4 = torch.nn.Linear(512, 256)
        self.linear_5 = torch.nn.Linear(256, 128)
        self.linear_6 = torch.nn.Linear(128, 32)
        self.score_layer = torch.nn.Linear(32, 4)
        self.num_sent=num_sent

        self.rank = torchsort.soft_rank



    def forward(self, input_ids, attn_mask):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """


        cat_list = []
 

        for i in range(self.num_sent):
            bert_out = self.bert_model(input_ids = input_ids[:,i,:].long(), attention_mask = attn_mask[:,i,:].float()).pooler_output
            shared_out = torch.relu(self.linear_1(bert_out))
            cat_list.append(shared_out)

        out = torch.cat(cat_list, 1)

        out = torch.relu(self.linear_2(out))


        out = torch.relu(self.linear_3(out))


        out = torch.relu(self.linear_4(out))


        out = torch.relu(self.linear_5(out))


        out = torch.relu(self.linear_6(out))
        out = self.score_layer(out)

        out = self.rank(out, regularization_strength=1.0) 

        return out