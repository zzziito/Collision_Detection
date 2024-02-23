import os,sys
import time
from pathlib import Path
from argparse import ArgumentParser
from tqdm.auto import tqdm

import yaml
import random, itertools
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Dict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.functional import kl_div, cross_entropy

from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn

from models import get_model
from data import get_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = "rnn"
batch_size = 1

dataset_kwargs = {
    'input_folder' : '/home/rtlink/robros/dataset/0215_norm/0215_free/input_data', 
    'target_folder': '/home/rtlink/robros/dataset/0215_norm/0215_free/target_data', 
    # 'collision_folder' : '/home/rtlink/robros/dataset/0215_dataset/collision',
    'num_joints' : 7,
    'seq_len': 1000, 
    'offset' : 1000
}

validset = get_dataloader(name=model_name, train=False, **dataset_kwargs)

loader_kwargs = dict(
    batch_size = batch_size,
    drop_last = True
)

valid_loader = DataLoader(validset, **loader_kwargs)

ckpt_path = '/home/rtlink/robros/log/0219/transformer/bs256_nhead8_nel6_lr6/ckpt.pt'
checkpoint = torch.load(ckpt_path)
model_CFG = checkpoint['cfg']


model_kwargs = dict(
    hidden_size=512, 
    num_joints=7, 
    num_layers=20,
    nhead=8,
    num_encoder_layers=6,
    )

model_name = "transformer"


model = get_model(model_name, **model_kwargs).cuda()
model.eval()


for _, input_data in zip(range(10), valid_loader):
    input_data = input_data[0].to(device) 
    _ = model(input_data)
 

num_tests = 100
times = []
 
with torch.no_grad():
    for _, input_data in zip(range(num_tests), valid_loader):
        input_data = input_data[0].to(device) 
        start_time = time.time()
        _ = model(input_data)
        end_time = time.time()
        times.append(end_time - start_time)
 
avg_time = sum(times) / len(times) * 1000  # 밀리초 단위로 변환
 
print(f"Average inference time: {avg_time:.2f} ms")