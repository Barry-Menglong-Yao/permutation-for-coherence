
import torch


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
    y = torchsort.soft_sort(x)
    print(y)

@lru_cache(None)
def cuda_toolkit_available():
    # https://github.com/idiap/fast-transformers/blob/master/setup.py
    try:
        call(["nvcc"], stdout=DEVNULL, stderr=DEVNULL)
        return True
    except FileNotFoundError:
        return False
 


print( cuda_toolkit_available())
test_sort()