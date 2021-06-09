
import torch
from torch.utils.data import random_split
import numpy as np
import os
import sys
from functools import lru_cache
from subprocess import DEVNULL, call
from dba.preprocess.preprocessor import *
from utils.enums import  *
from setuptools import setup
from torch.utils import cpp_extension
os.environ["CUDA_VISIBLE_DEVICES"]="3"
def test_sort():
    import torchsort
    
    x = torch.tensor([[8., 0., 5., 3., 2., 1., 6., 7., 9.]], requires_grad=True).to("cuda")
    # x = torch.tensor([[8., 0., 5., 3., 2., 1., 6., 7., 9.]], requires_grad=True) 
    y = torchsort.soft_sort(x.cpu())
    print(y)

@lru_cache(None)
def cuda_toolkit_available():
    # https://github.com/idiap/fast-transformers/blob/master/setup.py
    try:
        call(["nvcc"], stdout=DEVNULL, stderr=DEVNULL)
        return True
    except FileNotFoundError:
        return False
 

def variable_shape():
    x = np.array([[1., 2.], [3., 4.]])
    torch.from_numpy(x)
    print(x.dtype)
 
    y = np.array([[1., 2.], [3.]])
 
    device="cuda"
    from torchtext import data 
    ORDER = data.Field(batch_first=True, include_lengths=True, pad_token=0, use_vocab=False,
                        sequential=True)
    train_new_labels,label_length=ORDER.process(y, device=device)
     
 
     
    print(train_new_labels)

def gpu():
    cuda = torch.device('cuda')     # Default CUDA device
    cuda1 = torch.device('cuda:1')
    cuda3 = torch.device('cuda:3')  
    x = np.array([[1., 2.], [3., 4.]])
    x=torch.from_numpy(x)
    print(x)
    x.to('cuda:1')
    print(x)

def split_data():
    a,b,c=random_split(range(10), [3, 3,4] )
    print(len(a),len(b),len(c))



def split_str():
    mary = 'Mary had a little lamb'
    print( mary.split()) 

import zope.interface
class IFoo(zope.interface.Interface):
    x = zope.interface.Attribute("""X blah blah""")

    def bar(q, r=None):
        """bar blah blah"""


    
@zope.interface.implementer(IFoo)
class Foo:

    def __init__(self, x=None):
        self.x = x

    def bar(self, q, r=None):
        return q, r, self.x

    def __repr__(self):
        return "Foo(%s)" % self.x

def check_interface():
    print(IFoo.implementedBy(Foo))
    foo = Foo("In")
    print(foo.__repr__())

def dic_check():
    my_dict = {'name': 'Jack', 'age': 26}
    for key,metric in my_dict.items() :
        print(key+":"+str(metric)+"; ")
count_sentence_length(16,220,BertType.albert)
# dic_check()


 