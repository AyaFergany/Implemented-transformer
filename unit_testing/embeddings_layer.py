import sys
sys.path.append('')
from layers.embeddings import Embeddings

d_model = 4
vocab_size=6

# create the positional encoding matrix
pe = Embeddings(vocab_size, d_model)

# preview the values
print(pe.state_dict())