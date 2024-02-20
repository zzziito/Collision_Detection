import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
 
class Transformer(nn.Module):
    def __init__(self, num_joints, seq_len, hidden_size=512, nhead=8, num_encoder_layers=6, dropout=0.1):
        super(Transformer, self).__init__()
        self.num_joints = num_joints
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(hidden_size, dropout, max_len=seq_len)
        # Transformer Encoder Layer
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.input_linear = nn.Linear(3 * num_joints, hidden_size)
        self.output_linear = nn.Linear(hidden_size, num_joints)

    def forward(self, src):
        src = src.view(-1, self.seq_len, self.num_joints*3).permute(1,0,2)
        src = self.input_linear(src)  # [max_seq_len, batch_size, hidden_size]
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        output = self.output_linear(output)
        # 출력 차원 조정: [max_seq_len, batch_size, num_joints] -> [batch_size, num_joints, max_seq_len]
        output = output.permute(1, 2, 0)
        return output
 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
 
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000)/d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # pe = pe.repeat(1,3,1)

        self.register_buffer('pe', pe)
 
    def forward(self, x):
        pe = self.pe[:x.size(0), :]
        pe = pe.repeat(1, x.size(1) // pe.size(1), 1)
        x = x + pe
        return self.dropout(x)


    
if __name__=="__main__":

    num_joints = 7
    seq_len = 100
    hidden_size = 512
    nhead = 8
    num_encoder_layers = 6
    dropout = 0.1
    
    model = Transformer(num_joints=num_joints, seq_len=seq_len, 
                                    hidden_size=hidden_size, nhead=nhead,
                                    num_encoder_layers=num_encoder_layers, dropout=dropout)

    # if torch.cuda.is_available():
    #     model.cuda()

    batch_size = 4
    example_input = torch.randn(batch_size, num_joints, 3*100)
    
    # if torch.cuda.is_available():
    #     example_input = example_input.cuda()

    # with torch.no_grad():  
    output = model(example_input)
    
    print("Output tensor shape:", output.shape)