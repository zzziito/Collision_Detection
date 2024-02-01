import torch
import torch.nn as nn
import torch.nn.functional as F
 
class CNN(nn.Module):
    def __init__(self, num_joints, max_seq_len):
        super(CNN, self).__init__()
        self.num_joints = num_joints
        self.max_seq_len = max_seq_len

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), padding=(0, 1))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 3), padding=(0, 1))
 
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
 
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
 
        return x

if __name__=="__main__":

    num_joints = 7

    batch_size = 4
    max_seq_len = 100
    example_input = torch.randn(batch_size, 3, num_joints, max_seq_len)

    model = CNN(num_joints=num_joints, max_seq_len=max_seq_len)
    
    output = model(example_input)
    print("Output shape:", output.shape)