import torch
import torch.nn as nn
import torch.nn.functional as F
 
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size=100):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  
        return x
