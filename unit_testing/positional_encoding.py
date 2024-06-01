import sys
sys.path.append('')
from layers.positional_encoding import PositionalEncoding

d_model = 4
max_length = 10
dropout = 0.0

# create the positional encoding matrix
pe = PositionalEncoding(d_model, dropout, max_length)

# preview the values
print(pe.state_dict())


