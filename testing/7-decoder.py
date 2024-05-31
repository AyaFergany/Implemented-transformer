# importing required libraries
import math
import copy
import numpy as np

# torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
    """
    Args:
        d_model:      dimension of embeddings
        n_heads:      number of self attention heads
        dropout:      probability of dropout occurring
    """
    super().__init__()
    assert d_model % n_heads == 0            # ensure an even num of heads
    self.d_model = d_model                   # 512 dim
    self.n_heads = n_heads                   # 8 heads
    self.d_key = d_model // n_heads          # assume d_value equals d_key | 512/8=64

    self.Wq = nn.Linear(d_model, d_model)    # query weights
    self.Wk = nn.Linear(d_model, d_model)    # key weights
    self.Wv = nn.Linear(d_model, d_model)    # value weights
    self.Wo = nn.Linear(d_model, d_model)    # output weights

    self.dropout = nn.Dropout(p=dropout)     # initialize dropout layer  

  def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None):
    """
    Args:
       query:         query vector         (batch_size, q_length, d_model)
       key:           key vector           (batch_size, k_length, d_model)
       value:         value vector         (batch_size, s_length, d_model)
       mask:          mask for decoder     

    Returns:
       output:        attention values     (batch_size, q_length, d_model)
       attn_probs:    softmax scores       (batch_size, n_heads, q_length, k_length)
    """
    batch_size = key.size(0)                  
        
    # calculate query, key, and value tensors
    Q = self.Wq(query)                       # (32, 10, 512) x (512, 512) = (32, 10, 512)
    K = self.Wk(key)                         # (32, 10, 512) x (512, 512) = (32, 10, 512)
    V = self.Wv(value)                       # (32, 10, 512) x (512, 512) = (32, 10, 512)

    # split each tensor into n-heads to compute attention

    # query tensor
    Q = Q.view(batch_size,                   # (32, 10, 512) -> (32, 10, 8, 64) 
               -1,                           # -1 = q_length
               self.n_heads,              
               self.d_key
               ).permute(0, 2, 1, 3)         # (32, 10, 8, 64) -> (32, 8, 10, 64) = (batch_size, n_heads, q_length, d_key)
    # key tensor
    K = K.view(batch_size,                   # (32, 10, 512) -> (32, 10, 8, 64) 
               -1,                           # -1 = k_length
               self.n_heads,              
               self.d_key
               ).permute(0, 2, 1, 3)         # (32, 10, 8, 64) -> (32, 8, 10, 64) = (batch_size, n_heads, k_length, d_key)
    # value tensor
    V = V.view(batch_size,                   # (32, 10, 512) -> (32, 10, 8, 64) 
               -1,                           # -1 = v_length
               self.n_heads, 
               self.d_key
               ).permute(0, 2, 1, 3)         # (32, 10, 8, 64) -> (32, 8, 10, 64) = (batch_size, n_heads, v_length, d_key)
       
    # computes attention
    # scaled dot product -> QK^{T}
    scaled_dot_prod = torch.matmul(Q,        # (32, 8, 10, 64) x (32, 8, 64, 10) -> (32, 8, 10, 10) = (batch_size, n_heads, q_length, k_length)
                                   K.permute(0, 1, 3, 2)
                                   ) / math.sqrt(self.d_key)      # sqrt(64)
        
    # fill those positions of product as (-1e10) where mask positions are 0
    if mask is not None:
      scaled_dot_prod = scaled_dot_prod.masked_fill(mask == 0, -1e10)

    # apply softmax 
    attn_probs = torch.softmax(scaled_dot_prod, dim=-1)
        
    # multiply by values to get attention
    A = torch.matmul(self.dropout(attn_probs), V)       # (32, 8, 10, 10) x (32, 8, 10, 64) -> (32, 8, 10, 64)
                                                        # (batch_size, n_heads, q_length, k_length) x (batch_size, n_heads, v_length, d_key) -> (batch_size, n_heads, q_length, d_key)

    # reshape attention back to (32, 10, 512)
    A = A.permute(0, 2, 1, 3).contiguous()              # (32, 8, 10, 64) -> (32, 10, 8, 64)
    A = A.view(batch_size, -1, self.n_heads*self.d_key) # (32, 10, 8, 64) -> (32, 10, 8*64) -> (32, 10, 512) = (batch_size, q_length, d_model)
        
    # push through the final weight layer
    output = self.Wo(A)                                 # (32, 10, 512) x (512, 512) = (32, 10, 512) 

    return output, attn_probs                           # return attn_probs for visualization of the scores

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
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float=0.1):
        """
        Args: 
            d_model:  dimension of embeddings
            n_heads:  number of heads
            d_ffn:    dimension of feed-forward network
            dropout:  probability of dropout occuring
        """
        super().__init__()
      
        # masked multi-head attention sublayer
        self.masked_attention = MultiHeadAttention(d_model, n_heads, dropout)
        # layer norm for masked multi-head attention
        self.mask_attn_layer_norm = nn.LayerNorm(d_model)
        
        # multi-head attention sublayer
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        # layer norm for multi-head attention
        self.attn_layer_norm = nn.LayerNorm(d_model)
        
        # position-wise feed-forward network
        self.positionwise_ffn = PositionwiseFeedForward(d_model, d_ffn, dropout)
        # layer norm for position-wise ffn
        self.ffn_layer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg: Tensor, src: Tensor, trg_mask: Tensor, src_mask: Tensor):
       """
       Args:
          trg:        embedded sequences                (batch_size, trg_seq_length, d_model)
          src:        positionally embedded sequences   (batch_size, src_seq_length, d_model)
          trg_mask:   mask for the sequences            (batch_size, 1, 1, seq_lemgth)
          src_mask:   mask for the sequences            (batch_size, 1, 1, seq_length)
       Returns: 
          trg:        sequences after self-attention    (batch_size, trg_seq_length, d_model)
          attn_props: attention softmax scores

       """
       # pass trg embeddings through multi-head attention
       _trg, masked_atten_props = self.masked_attention(trg, trg, trg, trg_mask)
       
       # residual add and norm
       trg = self.masked_atten_layer_norm(trg + self.dropout(_trg))
       
       # pass trg and src embeddings through multi-head attention
       _trg, atten_props = self.attention(trg, src, src, src_mask)
       
       # residual add and norm
       trg = self.atten_layer_norm(trg + self.dropout(_trg))
      
       # position-wise feed-forward network
       _trg = self.positionwise_ffn(trg)

       # residual add and norm
       trg = self.positionwise_norm(trg + self.dropout(_trg))

       return trg, masked_atten_props, atten_props

class Decoder(nn.Module):
    def __init__(self, d_model: int, vocab_size: int,  n_heads: int, d_ffn: int, n_layers: int, dropout: float = 0.1):
      """
      Args:
         d_model:    dimension of embeddings
         vocab_size: size of the vocabulary
         n_heads:    number of heads
         d_ffn:      dimension of feed-forward network
         n_layers:   number of layers
         dropout:    probability of dropout occuring
       
      """
      super().__init__()
      # create n_layers encoders 
      self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ffn, dropout)
                                  for layer in range(n_layers)])
      
      self.dropout = nn.Dropout(dropout)

      # set output layer
      self.Wo = nn.Linear(d_model, vocab_size)
    
    def forward(self, trg: Tensor, src: Tensor, trg_mask: Tensor, src_mask: Tensor):
       """
       Args:
          trg:               mbedded sequences                 (batch_size, trg_seq_length, d_model)
          src:               embedded sequences                (batch_size, src_seq_length, d_model)
          trg_mask:          mask for te sequences             (batch_size, 1, 1, seq_length)
          src_mask:          mask for the sequences            (batch_size, 1, 1, seq_length)
       Returns: 
          trg:               sequences after self-attention    (batch_size, trg_seq_length, d_model)
          atten_props:       attention softmax scores
          masked_attn_props: masked attention softmax scores


       """
       # pass the sequences through each encoder
       for layer in self.layers:
          trg, masked_atten_props, atten_props = layer(trg, src, trg_mask, src_mask)

       self.masked_atten_props = masked_atten_props
       self.atten_props =  atten_props
       
       return self.Wo(trg)