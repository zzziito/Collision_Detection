import torch
import torch.nn as nn
import torch.nn.functional as F
import math
 
class Transformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, output_dim):
        super(Transformer, self).__init__()
 
        self.d_model = d_model

        # Input Embedding
        self.embed_src = nn.Linear(input_dim, d_model)
        self.embed_tgt = nn.Linear(input_dim, d_model)

        # positional encoder
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)

        # Transformer layer
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward)
        
        # 출력 데이터를 생성하기 위한 최종 선형 레이어
        self.    = nn.Linear(d_model, output_dim)
 
    def forward(self, src, tgt):
        # 소스와 타겟 데이터에 대해 임베딩 및 스케일 조정
        src = self.embed_src(src) * math.sqrt(self.d_model)
        tgt = self.embed_tgt(tgt) * math.sqrt(self.d_model)
 
        # positional encoding 적용
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        output = self.transformer(src, tgt)
        output = self.out(output)
 
        return output
 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # positional encoding 초기화
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        # 입력 데이터에 positional encoding 을 더해서 변환
        x = x + self.pe[:x.size(0), :]
        return x