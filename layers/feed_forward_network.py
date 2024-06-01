# importing required libraries
import math
import copy
import numpy as np

# torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float=0.1):
      """
      Args:
          d_model:   dimension of embeddings
          d_ffn:     dimension of feed-forward network
          dropout:   probability of droupout occuring
      """
      super().__init__()
      self.w_1 = nn.Linear(d_model, d_ffn)
      self.w_2 = nn.Linear(d_ffn, d_model)
      self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
       """
       Args:
           x:  output from attention (batch_size, seq_length, d_model)
        
       Returns:
           expanded-and-contracted representation (batch_size, seq_length, d_model)
       """
       return self.w_2(self.dropout(self.w_1(x).relu()))

