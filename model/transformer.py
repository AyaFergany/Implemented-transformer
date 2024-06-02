# importing required libraries
import math
import copy
import numpy as np
import sys

# torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.functional import pad

sys.path.append('')
from layers.embeddings import Embeddings
from layers.encoder import Encoder
from layers.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_embed: Embeddings, trg_embed: Embeddings, src_pad_idx: int, trg_pad_idx: int, device):
        """
        Args: 
           src_embed: source embeddings and encodings
           trg_embed: target embeddings and encodings
           src_pad_idx: padding index
           trg_pad_idx: padding index
           device:  cuda or cpu
        
        Returns: 
           output:  sequences after embedding
        """
        super().__init__()

        self.encoder: Encoder
        self.decoder: Decoder
        self.src_embed: src_embed
        self.trg_embed: trg_embed
        self.src_pad_idx: src_pad_idx
        self.trg_pad_idx: trg_pad_idx
        self.device: device

    def make_src_mask(self, src: Tensor):
        """
        Args:
           src:      raw sequences with padding   (batch_size, seq_length)

        Returns:
           src_mask: mask for each sequence       (batch_size, 1, 1, seq_length)
        
        """
        # assign 1 to tokens that need attended to and 0 to padding tokens, then add 2 dimensions
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask
    
    def make_trg_mask(self, trg: Tensor):
        """
        Args:  
           trg:       raw sequences with padding  (batch_size, seq_length)
        
        Returns:
           trg_mask:  mask for each sequence      (batch_size, 1, 1, seq_length)
        """

        seq_length = trg.shape[1]
        
        # assign True to tokens that need attended to and False to padding tokens, then add 2 dimensions
        trg_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        # generate subsequent mask
        trg_sub_mask = torch.trill(torch.ones(seq_length, seq_length), device=self.device)
            
        # bitwise "and" operator | 0 & 0 = 0, 1 & 1 = 1, 1 & 0 = 0
        trg_mask = trg_mask & trg_sub_mask

        return trg_mask
    
    def forward(self, src: Tensor, trg: Tensor):
     """
     Args: 
        src:    raw src sequences     (batch_size, src_seq_length)
        trg:    raw trg sequences     (batch_size, trg_seq_length)
      
     Returns:
         output:  sequnces after decoder   (batch_size, trg_seq_length, output_dim)
     """
     src_mask = self.make_src_mask(src)
     trg_mask = self.make_trg_mask(trg)

     src = self.encoder(self.src_embed, src_mask)

     output = self.decoder(self.trg_embed(trg), src, src_mask, trg_mask)

     return output


    
        
