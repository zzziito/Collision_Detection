import torch
import torch.nn as nn
 
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_joints, num_layers):
        super(RNN, self).__init__()
        self.num_joints = num_joints
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim * num_joints, output_dim)
 
    def forward(self, x):
        # [batch_size, num_joints, max_seq_len]
        batch_size, num_joints, seq_len = x.size()

        # 조인트별로 RNN에 입력하기 위해 차원 변경: [batch_size * num_joints, max_seq_len, input_dim]

        x = x.view(batch_size * num_joints, seq_len, -1)
        rnn_out, _ = self.rnn(x)
        rnn_out = rnn_out[:, -1, :]
        
        # 다시 [batch_size, num_joints * hidden_dim] 형태로 변형하여 모든 조인트 정보 통합
        rnn_out = rnn_out.view(batch_size, -1)

        output = self.fc(rnn_out)
        return output