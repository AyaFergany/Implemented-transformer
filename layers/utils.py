import torch
import spacy
import torchtext
from torch import Tensor
from torch.nn.functional import pad
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

example = "Hello! This is an example of a paragraph that has been split into its basic components. I wonder what will come next! Any guesses?"

def tokenize(sequence):
  # remove punctuation
  for punc in ["!", ".", "?"]:
    sequence = sequence.replace(punc, "")

  # split the sequence on spaces and lowercase each token
  return [token.lower() for token in sequence.split(" ")]

def build_vocab(data):
  # tokenize the data and remove duplicates
  vocab = list(set(tokenize(data)))

  # sort the vocabulary
  vocab.sort()

  # assign an integer to each word
  stoi = {word:i for i, word in enumerate(vocab)}

  return stoi

# build the vocab
stoi = build_vocab(example)

def sentence_to_indexes(vocab, sentence):
    """
    Converts a sentence (string) into a list of indices.
    """
    return [vocab[word] for word in sentence]

def make_src_mask(src: Tensor, pad_idx: int = 0):
  """
  Args:
      src:          raw sequences with padding        (batch_size, seq_length)              
    
  Returns:
      src_mask:     mask for each sequence            (batch_size, 1, 1, seq_length)
  """
  # assign 1 to tokens that need attended to and 0 to padding tokens, then add 2 dimensions
  src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

  return src_mask

def make_trg_mask(trg: Tensor, pad_idx: int = 0):
  """
  Args:
     trg:        raw sequences with padding    (batch_size, seq_length)

  Returns:
     trg_mask:   mask for each sequence        (batch_size, 1, seq_length, seq_length)
  """
  seq_length = trg.shape[1]

  # assign True to tokens that need attended to and False to padding tokens, then add 2 dimensions
  trg_mask = (trg != pad_idx).unsqueeze(1).unsqueeze(2)

  # generate subsequent mask
  trg_sub_mask = torch.tril(torch.ones((seq_length, seq_length))).bool()

  # bitwise "and" operator | 0 & 0 = 0, 1 & 1 = 1, 1 & 0 = 0
  trg_mask = trg_mask & trg_sub_mask

  return trg_mask

def pad_seq(seq: Tensor, max_length: int = 10, pad_idx: int = 0):
  """
  Args:
      seq:          raw sequence (batch_size, seq_length)
      max_length:   maximum length of a sequence
      pad_idx:      index for padding tokens           
    
  Returns:
      padded seq:   padded sequence (batch_size, max_length)
  """
  pad_to_add = max_length - len(seq) # amount of padding to add
  
  return pad(seq,(0, pad_to_add), value=pad_idx,)



