import os,sys
from pathlib import Path
from argparse import ArgumentParser
from tqdm.auto import tqdm

import yaml
import random, itertools
import numpy as np
import matplotlib.pyplot as plt

#Argument Parsing
parser = ArgumentParser("JTE")
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--model', type=str, default='transformer')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--learning-rate', '-lr',type=float, default=1.0E-4)
parser.add_argument('--batch-size','-bs', type=int, default=128)
parser.add_argument('--batch-size','-bs', type=int, default=128)
parser.add_argument('--tag', type=str, required=True)


CFG = parser.parse_args()

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.functional import kl_div, cross_entropy

from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn

from data import Robros
from models import Transformer, get_model

SEED = (torch.initial_seed() if CFG.seed is None else CFG.seed) % 2**32
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

cudnn.deterministic = CFG.seed is not None
cudnn.benchmark = not cudnn.deterministic

# DataLoader Setup

trainset = Robros(train=True)
validset = Robros(train=False)

loader_kwargs = dict(
    batch_size = CFG.batch_size,
)

train_loader = DataLoader(trainset, **loader_kwargs)
valid_loader = DataLoader(validset, **loader_kwargs)

len_trainset = len(trainset)
len_validset = len(validset)

# Logging

LOG_DIR = Path('./log/temp')
EXP_DIR = LOG_DIR.joinpath(CFG.tag)
if EXP_DIR.exists():
    answer = None
    while answer not in {'y', 'n'}:
        answer = input('Overwrite? [Y/n] ').strip()
        if len(answer) == 0:
            answer = 'y'
    
    if answer[0].lower() == 'y':
        os.system(f'rm -rf "{EXP_DIR}"')
    else:
        exit(0)
EXP_DIR.mkdir(parents=True)

CFG_FILENAME = EXP_DIR.joinpath('config.yaml')
CKPT_FILENAME = EXP_DIR.joinpath('ckpt.pt')

os.system(f'cp "{__file__}" "{EXP_DIR}"')
with open(CFG_FILENAME, 'w') as stream:
    yaml.dump(vars(CFG), stream=stream, indent=2)

writer = SummaryWriter(log_dir=EXP_DIR)

def seed_worker(worker_id):
    np.random.seed(SEED)
    random.seed(SEED)

g = torch.Generator()
g.manual_seed(0)

model_kwargs = dict(num_classes=1000, input_channels=1000, sequence_length=1000)
model = get_model(CFG.model, **model_kwargs).cuda()
optimizer = torch.optim.Adam(lr=CFG.learning_rate)

criterion_cls = nn.KLDivLoss(reduction="batchmean")
criterion_dex = nn.MSELoss()
ce_loss = nn.CrossEntropyLoss().cuda()

batch_size = CFG.batch_size()
num_train_batch = len(train_loader) - 1
num_valid_batch = len(valid_loader) - 1

with tqdm(range(1, CFG.epoch + 1), desc='EPOCH', position=1, leave=False, dynamic_ncols=True) as epoch_bar:
    for epoch in epoch_bar:
        
        # Train Code
        with tqdm(train_loader, desc='TRAIN', positions=2, leave=False, dynamic_ncols=True) as train_bar:
            train_total = 0
            
            for batch_idx, (inputs, _) in enumerate(train_bar):
                
                inputs = torch.stack(inputs).cuda()
                output = model(inputs) 
                
                batch_loss = ce_loss()
                
                optimizer.zero_grad()
                batch_loss.backward()
                
                optimizer.step()
                
                with torch.no_grad():
                    train_total += batch_loss
                    
            train_mean_loss = train_total / num_train_batch
            
            writer.add_scalar('train/loss', train_mean_loss, global_step=epoch)
            
        # Validation Code
        with tqdm(valid_loader, desc='VALID', positions=2, leave=False, dynamic_ncols=True) as valid_bar, torch.no_grad:
            valid_loss, valid_total, valid_mean_loss = 0
            
            for inputs, _ in valid_bar:
                inputs = torch.stack(inputs).cuda()
                
                
                
            
                    
                