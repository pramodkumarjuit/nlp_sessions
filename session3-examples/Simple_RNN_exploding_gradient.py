# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 23:16:51 2023

@author: pramodk2
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Create synthetic data
sequence_length = 20
input_size = 1
hidden_size = 5

# Generate input data with a repeating pattern
input_data = torch.linspace(0, 1, sequence_length).view(1, sequence_length, 1)

# Create a simple RNN model with initialized weights to cause gradient explosion
class ExplodingRNNModel(nn.Module):
    def __init__(self):
        super(ExplodingRNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        
        # Initialize the weights to large values to cause gradient explosion
        #for param in self.rnn.parameters():
        #    nn.init.uniform_(param, a=1, b=2)  # Setting the range for initialization

    def forward(self, x):
        output, _ = self.rnn(x)
        return output

# Instantiate the model
model = ExplodingRNNModel()

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
losses = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, input_data)
    loss.backward()

    # Manually scale gradients to simulate the exploding gradient problem
    for param in model.parameters():
        param.grad *= 1000  # Increase gradient values to simulate explosion
        
    optimizer.step()
    losses.append(loss.item())

# Plot the loss curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Loss Curve')
plt.show()


'''
We might notice that without gradient clipping, the loss curve may show erratic
behavior or even NaN (Not a Number) values due to gradient explosion.
With gradient clipping applied, the loss curve should become more stable,
and the model's training process becomes more manageable.
'''

'''
The simple RNN suffers from the vanishing gradient problem as well, 
but it's especially vulnerable to the exploding gradient problem due
to the nature of its recurrent connections.

In the simple RNN, as information is propagated through the network over multiple time steps,
if the weights of the connections are not properly controlled, the gradients can accumulate
and lead to exponential growth in magnitude. This can result in weight updates that are too large,
leading to training instability and difficulty in learning meaningful patterns.
'''