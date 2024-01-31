import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, num_joints, embed_dim):
        super(RNN, self).__init__()
        
        # Joint embedding 레이어
        self.embed_joint = nn.Embedding(num_joints, embed_dim)
        self.rnn = nn.RNN(input_dim + embed_dim, hidden_dim, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, joint_number):
        
        joint_number = joint_number.long() 
        joint_embedding = self.embed_joint(joint_number)
        
        # 데이터와 joint 임베딩 결합
        x = torch.cat((x, joint_embedding), dim=2)
        
        output, hidden = self.rnn(x)
        output = self.out(output[:, -1, :])  # 시퀀스의 마지막 출력만 사용

        return output

