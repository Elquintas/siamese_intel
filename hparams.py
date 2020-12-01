import numpy as np

lr = 0.0001    #0.001
epochs = 1   #24    > <
batch_size = 64   #128

# Amount of GRU layers in the encoder
n_gru_layers = 2 

# hidden dimension of the encoder
# the embedding size corresponds to
# hidden_dim*2 in the case of bidirectional
# GRUs. If not, embedding size = hidden_dim
hidden_dim = 100 
bidirectional=True

# Dropout rate of the siamese encoder (GRU)
rnn_dropout = 0.0 

# Dropout rate of the DNN part of the model
dropout = 0.0

# Dimension of the features being used, 
#in this case 13 MFCCs.
# other examples:
#  ---> 40 filterbanks
#  ---> 35 phonetic posteriors
feats_dim=13

checkpoint_interval = 1
sigmoid_threshold = 0.5
