
import torch

import numpy as np
import os
import sys
from functools import lru_cache
from subprocess import DEVNULL, call

 
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


variable_shape()