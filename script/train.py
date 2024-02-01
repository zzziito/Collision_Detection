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
parser.add_argument('--model', type=str, default='rnn')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--learning-rate', '-lr',type=float, default=1.0E-4)
parser.add_argument('--batch-size','-bs', type=int, default=128)
parser.add_argument('--tag', type=str, required=True)

# Arguments for RNN
parser.add_argument('--num-layers', '--nl', type=int, required=False)
parser.add_argument('--hidden-size', '--hs', type=int, required=False)



CFG = parser.parse_args()

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.functional import kl_div, cross_entropy

from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn

from models import get_model
from data import get_dataloader

SEED = (torch.initial_seed() if CFG.seed is None else CFG.seed) % 2**32
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

cudnn.deterministic = CFG.seed is not None
cudnn.benchmark = not cudnn.deterministic

# CUDA setup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DataLoader Setup

model_name = CFG.model

dataset_kwargs = {
    'input_folder_path' : "/home/rtlink/robros/dataset/robros_dataset/input_data",
    'target_folder_path': "/home/rtlink/robros/dataset/robros_dataset/target_data",
    'num_joints' : 7,
}

trainset = get_dataloader(name=model_name, train=True, **dataset_kwargs)
validset = get_dataloader(name=model_name, train=False, **dataset_kwargs)

loader_kwargs = dict(
    batch_size = CFG.batch_size,
    drop_last = True
)

train_loader = DataLoader(trainset, **loader_kwargs)
valid_loader = DataLoader(validset, **loader_kwargs)

len_trainset = len(trainset)
len_validset = len(validset)

# Logging

LOG_DIR = Path('./log/RNN')
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

model_kwargs = dict(
    hidden_size=CFG.hidden_size, 
    num_joints=7, 
    num_layers=CFG.num_layers
    )

model = get_model(CFG.model, **model_kwargs).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate)

criterion_cls = nn.KLDivLoss(reduction="batchmean")
criterion_dex = nn.MSELoss()
ce_loss = nn.CrossEntropyLoss().cuda()

batch_size = CFG.batch_size
num_train_batch = len(train_loader) - 1
num_valid_batch = len(valid_loader) - 1

with tqdm(range(1, CFG.epoch + 1), desc='EPOCH', position=1, leave=False, dynamic_ncols=True) as epoch_bar:
    lr_list, loss_list = [], []

    for epoch in epoch_bar:
        # Train Code
        with tqdm(train_loader, desc='TRAIN', position=2, leave=False, dynamic_ncols=True) as train_bar:
            train_loss, train_total, train_mean_loss = 0, 0, 0
            
            for batch_idx, (input, target) in enumerate(train_bar):
                
                input, target = input.to(device), target.to(device)

                optimizer.zero_grad()

                output = model(input)

                log_softmax_output = F.log_softmax(output, dim=-1)
                target_dist = F.softmax(target, dim=-1)

                output_shape = log_softmax_output.shape
                target_shape = target_dist.shape

                loss = F.kl_div(log_softmax_output, target_dist, reduction='batchmean')

                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    train_loss += loss.item()*input.size(0)
                    train_total += input.size(0)
                    
            train_mean_loss = train_loss / train_total
            
            writer.add_scalar('train/loss', train_mean_loss, global_step=epoch)
            
        # Validation Code
        with tqdm(valid_loader, desc='VALID', position=2, leave=False, dynamic_ncols=True) as valid_bar, torch.no_grad():
            valid_loss, valid_total, valid_mean_loss = 0, 0, 0

            for batch_idx, (input, target) in enumerate(valid_bar):

                input, target = input.to(device), target.to(device)

                outputs = model(input)

                log_softmax_output = F.log_softmax(output, dim=-1)
                target_dist = F.softmax(target, dim=-1)



                loss = F.kl_div(log_softmax_output, target_dist, reduction='batchmean')

                valid_loss += loss.item()*input.size(0)
                valid_total += input.size(0)

            valid_mean_loss = valid_loss / valid_total
            writer.add_scalar('valid/loss', valid_mean_loss, global_step=epoch)

        lr_list.append(optimizer.param_groups[0]['lr'])
        loss_list.append(train_mean_loss)

        
        torch.save({
            'cfg': CFG,
            'last_epoch': epoch,
            'lr_list': lr_list,
            'loss_list': loss_list,
            'saved_hidden_size': CFG.hidden_size,
            'saved_num_layers': CFG.num_layers,
            'saved_epoch': CFG.epoch,
            'saved_learning_rate': CFG.learning_rate,
            'saved_batch_size': CFG.batch_size,
        }, CKPT_FILENAME)
 
        writer.add_scalar('epoch/epoch', epoch, global_step=epoch)
        writer.add_scalar('epoch/learning_rate', lr_list[-1], global_step=epoch)





                
                
                
            
                    
                