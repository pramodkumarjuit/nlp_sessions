# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 03:07:25 2023

@author: pramodk kumar
"""

# PyTorch example that demonstrates the use of a simple RNN layer:
    
import torch
import torch.nn as nn

# Define the parameters
input_size = 5      # each time step has 5 elements inputs
hidden_size = 10
sequence_length = 3 # length of the input sequence (RNN process one element at a time of input sequences)
num_layers = 1

# from pytorch doc: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
#    num_layers – Number of recurrent layers.
#       E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN,
#       with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1
    
# Create a simple RNN layer
rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

# Generate synthetic input data
batch_size = 2
input_data = torch.randn(batch_size, sequence_length, input_size)
print(f'Input Shape: {input_data.shape}')

# Initialize the hidden state
hidden = torch.zeros(num_layers, batch_size, hidden_size)

# Forward pass through the RNN
#   - The output tensor contains the output of the RNN at each time step for each sequence in the batch.
#       - output shape = (batch_size, sequence_length, hidden_size)    
#   - The hidden state tensor represents the state of RNN after processing the entire sequence in the batch
output, new_hidden = rnn(input_data, hidden)



print(f'Output Shape: {output.shape}')
print(f'Hidden State Shape: {new_hidden.shape}')
print(50*'=')



# Print the output and new hidden state
print("Input:", input_data)
print("Output:", output)
print("New Hidden State:", new_hidden)

# Notes:
#   - An RNN produces an output after processing a sequence. 

