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
from layers.encoder import Encoder
from layers.decoder import Decoder
from layers.utils import tokenize, build_vocab, make_src_mask, pad_seq, sentence_to_indexes, make_trg_mask, stoi

# Training a Simple Model
de_example = "Hallo! Dies ist ein Beispiel für einen Absatz, der in seine Grundkomponenten aufgeteilt wurde. Ich frage mich, was als nächstes kommt! Irgendwelche Ideen?"
en_example = "Hello! This is an example of a paragraph that has been split into its basic components. I wonder what will come next! Any guesses?"

# build the vocab
de_stoi = build_vocab(de_example)
en_stoi = build_vocab(en_example)

# build integer-to-string decoder for the vocab
de_itos = {v:k for k,v in de_stoi.items()}
en_itos = {v:k for k,v in en_stoi.items()}

sequences = ['What will come next?',
             'This is a basic paragraph.',
             'A basic split will come next!']

# tokenize the sequences
tokenized_sequences = [tokenize(seq) for seq in sequences]

# index the sequences
indexed_sequences = [[stoi[word] for word in seq] for seq in tokenized_sequences]

max_length = 8
pad_idx = len(stoi)

padded_seqs = []

for seq in indexed_sequences:
  # pad each sequence
  padded_seqs.append(pad_seq(torch.Tensor(seq), max_length, pad_idx))

# create a tensor from the padded sequences
tensor_sequences = torch.stack(padded_seqs).long()

# create the source masks for the sequences
src_mask = make_src_mask(tensor_sequences, pad_idx)

de_sequences = ["Hallo! Dies ist ein Beispiel für einen Absatz, der in seine Grundkomponenten aufgeteilt wurde. Ich frage mich, was als nächstes kommt! Irgendwelche Ideen?"]
en_sequences = ["Hello! This is an example of a paragraph that has been split into its basic components. I wonder what will come next! Any guesses?"]

def sentence_to_indexes(vocab, sentence):
    """
    Converts a sentence (string) into a list of indices.
    """
    return [vocab[word] for word in sentence]

# Tokenize the sentences
de_tokenized_sequences = [tokenize(s) for s in de_sequences]
en_tokenized_sequences = [tokenize(s) for s in en_sequences]

# Convert sentences to sequences of integers
de_indexed_sequences = [sentence_to_indexes(de_stoi, s) for s in de_tokenized_sequences]
en_indexed_sequences = [sentence_to_indexes(en_stoi, s) for s in en_tokenized_sequences]

# Add '<pad>' to the dictionaries
de_stoi['<pad>'] = len(de_stoi)
en_stoi['<pad>'] = len(en_stoi)

# pad the sequences
max_length = 9
pad_idx = de_stoi['<pad>']

de_padded_seqs = []
en_padded_seqs = []

# pad each sequence
for de_seq, en_seq in zip(de_indexed_sequences, en_indexed_sequences):
    de_padded_seqs.append(pad_seq(torch.Tensor(de_seq), max_length, pad_idx))
    en_padded_seqs.append(pad_seq(torch.Tensor(en_seq), max_length, pad_idx))

# create a tensor from the padded sequences
de_tensor_sequences = torch.stack(de_padded_seqs).long()
en_tensor_sequences = torch.stack(en_padded_seqs).long()

# remove last token
trg = en_tensor_sequences[:,:-1]

# remove the first token
expected_output = en_tensor_sequences[:,1:]

# generate masks
src_mask = make_src_mask(de_tensor_sequences, pad_idx)
trg_mask = make_trg_mask(trg, pad_idx)

# parameters
de_vocab_size = len(de_stoi)
en_vocab_size = len(en_stoi)
d_model = 32
n_heads = 4
d_ffn = d_model * 4
n_layers = 3
dropout = 0.1
max_pe_length = 10

# create the embeddings
de_lut = Embeddings(de_vocab_size, d_model)
en_lut = Embeddings(en_vocab_size, d_model)

# create the positional encodings
pe = PositionalEncoding(d_model=d_model, dropout=0.1, max_length=max_pe_length)

# embed and encode
de_embedded = nn.Sequential(de_lut, pe)
en_embedded = nn.Sequential(en_lut, pe)

# initialize encoder
encoder = Encoder(d_model, n_heads, d_ffn, n_layers, dropout)

# initialize the decoder
decoder = Decoder(d_model, en_vocab_size, n_heads, d_ffn, n_layers, dropout)

# initialize the model
model = nn.ModuleList([de_embedded, en_embedded, encoder, decoder])

# normalize the weights
for p in model.parameters():
  if p.dim() > 1:
    nn.init.xavier_uniform_(p)

# hyperparameters
LEARNING_RATE = 0.005
EPOCHS = 50

# adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

# loss function
criterion = nn.CrossEntropyLoss(ignore_index= en_stoi["<pad>"])

# set the model to training mode
model.train()

# loop through each epoch
for i in range(EPOCHS):
  epoch_loss = 0

  # zero the gradients
  optimizer.zero_grad()

  # pass through encoder
  encoded_embeddings = encoder(src=de_embedded(de_tensor_sequences),
                               src_mask=src_mask)

  # logits for each output
  logits = decoder(trg=en_embedded(trg), src=encoded_embeddings,
                   trg_mask=trg_mask,
                   src_mask=src_mask)

  # calculate the loss
  loss = criterion(logits.contiguous().view(-1, logits.shape[-1]),
                   expected_output.contiguous().view(-1))

  # backpropagation
  loss.backward()

  # clip the weights
  torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

  # update the weights
  optimizer.step()

  # preview the predictions
  predictions = [[en_itos[tok] for tok in seq] for seq in logits.argmax(-1).tolist()]

  if i % 7 == 0:
    print("="*25)
    print(f"epoch: {i}")
    print(f"loss: {loss.item()}")
    print(f"predictions: {predictions}")