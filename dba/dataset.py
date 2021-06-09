from torch.utils.data import Dataset 
import torch
class NipsDataset(Dataset):

    def __init__(self, text,sentence_num_list,labels, tokenizer,max_len ):
        self.tokenizer = tokenizer
        self.text = text#dataframe['abstract']
        self.labels = labels
        self.max_len=max_len
        self.sentence_num_list=sentence_num_list

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        sent_num=self.sentence_num_list[index]
        text_list=self.text[index]
        inputs=self.tokenizer(text_list, return_tensors="pt", padding='max_length',truncation =True,max_length=self.max_len ) 
        ids = inputs['input_ids']
        mask = inputs['attention_mask'] 
        mask[sent_num:,:]=True


        labels=torch.tensor(self.labels[index], dtype=torch.float)
        sent_num=torch.tensor(sent_num, dtype=torch.long)
        return  ids, mask,sent_num,labels
            
            
         