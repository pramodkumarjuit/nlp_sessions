# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 22:44:27 2023

@author: pramodk2
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = '1'

# Create synthetic data
sequence_length = 20
input_size = 1
hidden_size = 5

# Generate input data with a repeating pattern
input_data = torch.sin(torch.linspace(0, 10, sequence_length)).view(1, sequence_length, 1)

# Create a simple RNN model
class SimpleRNNModel(nn.Module):
    def __init__(self):
        super(SimpleRNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x):
        output, _ = self.rnn(x)
        return output

# Instantiate the model
model = SimpleRNNModel()

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 3000
losses = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, input_data)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Plot the loss curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Loss Curve')
plt.show()

'''
We might notice that the loss curve in the plot starts decreasing 
initially but then plateaus or decreases very slowly.
This is a typical behavior associated with the vanishing gradient problem.
'''
