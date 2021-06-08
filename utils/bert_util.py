from transformers import AlbertTokenizer
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import  AlbertModel
from utils.enums import  *
def gen_tokenizer( bert_type):
    if bert_type==BertType.distilbert:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    else:
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    return tokenizer



def gen_bert_model(bert_type):
    if bert_type==BertType.distilbert:
        bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    else:
        bert_model = AlbertModel.from_pretrained('albert-base-v2') 
    return bert_model


