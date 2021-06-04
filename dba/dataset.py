from torch.utils.data import Dataset 
import torch
class NipsDataset(Dataset):

    def __init__(self, dataframe,labels, tokenizer ):
        self.tokenizer = tokenizer
        self.abstract = dataframe['abstract']
        self.labels = labels
 

    def __len__(self):
        return len(self.abstract)

    def __getitem__(self, index):
        comment_text = str(self.abstract[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.labels[index], dtype=torch.float)
        }