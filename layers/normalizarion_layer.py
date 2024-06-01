# importing required libraries
import math
import copy
import numpy as np

# torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class LayerNorm(nn.Module):
  def __init__(self, features, eps=1e-5):
    super().__init__()
    # initialize gamma to be all ones
    self.gamma = nn.Parameter(torch.ones(features)) 
    # initialize beta to be all zeros
    self.beta = nn.Parameter(torch.zeros(features)) 
    # initialize epsilon
    self.eps = eps

  def forward(self, src):
    # mean of the token embeddings
    mean = src.mean(-1, keepdim=True)        
    # variance of the token embeddings         
    var = src.var(-1, keepdim=True,unbiased=False)  
    # return the normalized value  
    return self.gamma * (src - mean) / torch.sqrt(var + self.eps) + self.beta 
