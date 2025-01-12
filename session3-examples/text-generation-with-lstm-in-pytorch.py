# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 01:38:34 2023

@author: pramodk2
"""

"""
A simple model using LSTM layer for single char prediction.
It is traing using a a texbook (Alice’s Adventures in Wonderland ) data.
A window of 100 chars are used, meaning on giving 100 chars model tries to
predicts 101 char.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
 
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
 
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)


 # A window of 100 character is used here. 
 # That is, with character 1 to 100 as input,  the model is going to predict for character 101

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
total_seq =  n_chars - seq_length
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)
print("Total Patterns: ", n_patterns) # 160872
print("Total seq: ", total_seq)       # 160872

#
# reshape X to be [samples, time steps, features]  for LATM layer compatiable
# The input is single feature (i.e., one integer for one character).
X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, seq_length, 1)
X = X / float(n_vocab)
y = torch.tensor(dataY)
print(X.shape, y.shape)
print('First Sequence:', X[0])

# Prepare a Model
#   hidden_size, num_layers - hyperparameter
#   input_size: The input is single feature hence 1
#   At each time step has a shape (1, 100, 1) ==> (batch, seq_length, input_size)
class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, n_vocab)
    def forward(self, x):
        y, _ = self.lstm(x) # Final hidden state & Final o/p are same
        # take only the last output
        y = y[:, -1, :]
        # produce output
        y = self.linear(self.dropout(y))
        return y
    
# Training...
n_epochs = 4 #40
batch_size = 128
model = CharModel()
 
optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss(reduction="sum")
loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)
 
best_model = None
best_loss = np.inf
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader: # X_batch Size: torch.Size([128, 100, 1]); y_batch Size: torch.Size([128])
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss += loss_fn(y_pred, y_batch)
        if loss < best_loss:
            best_loss = loss
            best_model = model.state_dict()
        print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))
 
torch.save([best_model, char_to_int], "single-char.pth")


############ Inference #############
best_model, char_to_int = torch.load("single-char.pth")
n_vocab = len(char_to_int)
int_to_char = dict((i, c) for c, i in char_to_int.items())
 
#reload the model
class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, n_vocab)
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x

model = CharModel()
model.load_state_dict(best_model)
 
# randomly generate a prompt
filename = "wonderland.txt"
seq_length = 100
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
start = np.random.randint(0, len(raw_text)-seq_length)
prompt = raw_text[start:start+seq_length]
pattern = [char_to_int[c] for c in prompt]
 
model.eval()
print('Prompt: "%s"' % prompt)
with torch.no_grad():
    for i in range(1000):
        # format input array of int into PyTorch tensor
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        x = torch.tensor(x, dtype=torch.float32)
        # generate logits as output from the model
        prediction = model(x)
        # convert logits into one character
        index = int(prediction.argmax())
        result = int_to_char[index]
        print(result, end="") # print output without a new line
        # append the new character into the prompt for the next iteration
        pattern.append(index)
        pattern = pattern[1:]
print()
print("Done.")
