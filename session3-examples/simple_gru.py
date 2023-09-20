# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 07:06:39 2023

@author: pramodk2
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Sample text data and labels
texts = ["This is a positive review.", "This is a negative review.", "A positive sentiment.", "Negative feedback."]
labels = [1, 0, 1, 0]  # 1: Positive, 0: Negative

# Tokenize and vectorize text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()
y = torch.tensor(labels, dtype=torch.long)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), y_train)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Hyperparameters
n_layers = 1

# Define GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, n_layers = 1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # GRU only has a single hidden state
        # QUIZ: gru(x,h) ==> but here only input is provide!!!
        out, hidden = self.gru(x)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)  # Take the output of the last time step
        return out

# Hyperparameters
input_size = X_train.shape[1]  # Input features (vocabulary size)
hidden_size = 16
num_classes = 2  # Number of classes

# Instantiate the model
model = GRUModel(input_size, hidden_size, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    test_outputs = model(torch.tensor(X_test, dtype=torch.float32))
    predicted_labels = torch.argmax(test_outputs, dim=1)
    accuracy = (predicted_labels - y_test).sum().item() / y_test.size(0)
    print(f"Test Accuracy: {accuracy:.4f}")
