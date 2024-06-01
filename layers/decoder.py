# importing required libraries
import math
import copy
import numpy as np

# torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from layers.multi_head_attention import MultiHeadAttention
from layers.feed_forward_network import PositionwiseFeedForward
    
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
       trg = self.mask_attn_layer_norm(trg + self.dropout(_trg))
       
       # pass trg and src embeddings through multi-head attention
       _trg, atten_props = self.attention(trg, src, src, src_mask)
       
       # residual add and norm
       trg = self.attn_layer_norm(trg + self.dropout(_trg))
      
       # position-wise feed-forward network
       _trg = self.positionwise_ffn(trg)

       # residual add and norm
       trg = self.positionwise_ffn(trg + self.dropout(_trg))

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