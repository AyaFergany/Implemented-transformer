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
from layers.positional_encoding import PositionalEncoding
from layers.multi_head_attention import MultiHeadAttention
from layers.feed_forward_network import PositionwiseFeedForward
from layers.utils import tokenize, stoi

torch.set_printoptions(precision=2, sci_mode=False)

# convert the sequences to integers
sequences = ["I wonder what will come next!",
             "This is a basic example paragraph.",
             "Hello what is a basic split?"]

# tokenize the sequences
tokenized_sequences = [tokenize(seq) for seq in sequences]

# index the sequences
indexed_sequences = [[stoi[word] for word in seq] for seq in tokenized_sequences] 

# convert the sequences to tensor
tensor_sequences = torch.tensor(indexed_sequences).long()

# vocab_size
vocab_size = len(stoi)

# embedding dimension
d_model = 8

# create the embeddings
lut = Embeddings(vocab_size, d_model)

# create the positional encodings
pe = PositionalEncoding(d_model=d_model, dropout=0.1, max_length=10)

# embed the sequence
embeddings = lut(tensor_sequences)

# positionally encode the sequences
X = pe(embeddings)

# set the n_heads
n_heads = 4

# create the attention layer
attention = MultiHeadAttention(d_model, n_heads, dropout=0.1)

# pass X through the attention layer three times to create Q, K and V
output, atten_props = attention(X, X, X, mask=None)

# calculate the d_ffn
d_ffn = d_model * 4

# pass the tensor through the position-wise feed-forward network
fnn = PositionwiseFeedForward(d_model, d_ffn, dropout=0.1)

print(fnn(output))