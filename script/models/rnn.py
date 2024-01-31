import torch
import torch.nn as nn
 
class RNN(nn.Module):
    def __init__(self, num_joints, hidden_size, num_layers=10):
        super(RNN, self).__init__()
        self.num_joints = num_joints
        self.hidden_dim = hidden_size
        self.num_layers = num_layers

        input_size = 3*num_joints
        output_size = num_joints

        self.rnn = nn.RNN(input_size=input_size, 
                          hidden_size=hidden_size, 
                          num_layers=num_layers, 
                          batch_first=True)
        
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):

        x = x.transpose(1,2)

        out, _ = self.rnn(x)
        out = self.fc(out)
        out = out.transpose(1,2)

        return out