import os,sys
from pathlib import Path
from argparse import ArgumentParser
from tqdm.auto import tqdm

import yaml
import random, itertools
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Dict

#Argument Parsing
parser = ArgumentParser("TCD")
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--model', type=str, default='rnn')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--learning-rate', '-lr',type=float, default=1.0E-4)
parser.add_argument('--batch-size','-bs', type=int, default=128)
parser.add_argument('--tag', type=str, required=True)

# Arguments for RNN
parser.add_argument('--num-layers', '--nl', type=int, required=False)
parser.add_argument('--hidden-size', '--hs', type=int, required=False)

# Arguments for Transformer
parser.add_argument('--nhead', type=int, required=False)
parser.add_argument('--num-encoder-layers', '--nel', type=int, required=False)

# Arguments for Discriminator


CFG = parser.parse_args()

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.functional import kl_div, cross_entropy

from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn

from models import get_model
from models.discriminator_tf import Discriminator
# from models.discriminator_rn import Discriminator
# from models.discriminator_fc import Discriminator
from data import get_dataloader


SEED = (torch.initial_seed() if CFG.seed is None else CFG.seed) % 2**32
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

cudnn.deterministic = CFG.seed is not None
cudnn.benchmark = not cudnn.deterministic

### CUDA setup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### DataLoader Setup

model_name = CFG.model

dataset_kwargs = {
    'input_folder' : '/home/rtlink/robros/dataset/0215_norm/0215_collision/input_data', 
    'target_folder': '/home/rtlink/robros/dataset/0215_norm/0215_collision/target_data', 
    'collision_folder' : '/home/rtlink/robros/dataset/0215_norm/0215_collision/collision',
    'num_joints' : 7,
    'seq_len': 3000, 
    'offset' : 3000
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


### Logging

LOG_DIR = Path('./log/0221/discriminator')
MODEL_DIR = LOG_DIR.joinpath(CFG.model) 
EXP_DIR = MODEL_DIR.joinpath(CFG.tag)

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

### Torque Estimator

# ckpt_path = '/home/rtlink/robros/log/0219/rnn/bs256_hs50_nl10_lr6/ckpt.pt'
# checkpoint = torch.load(ckpt_path)
# model_CFG = checkpoint['cfg']

# model_kwargs = dict(
#     hidden_size=CFG.hidden_size, 
#     num_joints=7, 
#     num_layers=CFG.num_layers,
#     nhead=CFG.nhead,
#     num_encoder_layers=CFG.num_encoder_layers,
#     )

# model = get_model(CFG.model, **model_kwargs).cuda()
# model.eval()

def clone_state_dict(thing: Union[nn.Module, Dict[str, torch.Tensor]]):
    if isinstance(thing, nn.Module):
        state_dict = thing.state_dict()
    elif isinstance(thing, dict):
        state_dict = thing
    else:
        raise TypeError(f"Expected `nn.Module` or `dict[str, torch.Tensor]` for `thing` but got `{repr(thing)}` instead.")
    
    return {key: val.clone().detach().cpu() for key, val in state_dict.items()}

### Collision Discriminator

# collision_discriminator = Discriminator(input_size=3000, hidden_size1=64, hidden_size2=32, output_size=3000).cuda()
collision_discriminator = Discriminator(input_size=3000, hidden_size=512, nhead=8, num_encoder_layers=6).cuda()
# collision_discriminator = Discriminator(input_size=3000, hidden_size=100, num_layers=32).cuda()

###

optimizer = torch.optim.Adam(collision_discriminator.parameters(), lr=CFG.learning_rate)

criterion_cls = nn.KLDivLoss(reduction="batchmean")
criterion_dex = nn.MSELoss()
ce_loss = nn.CrossEntropyLoss().cuda()

batch_size = CFG.batch_size
num_train_batch = len(train_loader) - 1
num_valid_batch = len(valid_loader) - 1

with tqdm(range(1, CFG.epoch + 1), desc='EPOCH', position=1, leave=False, dynamic_ncols=True) as epoch_bar:
    lr_list, loss_list, acc_list = [], [], []

    for epoch in epoch_bar:
        # Train Code
        with tqdm(train_loader, desc='TRAIN', position=2, leave=False, dynamic_ncols=True) as train_bar:
            train_loss, train_total, train_accuracy = 0, 0, 0
            train_mean_loss, train_mean_acc = 0,0,
            
            for batch_idx, (_, target, collision) in enumerate(train_bar):
                
                _, target, collision = _, target.to(device), collision.to(device)

                with torch.no_grad():
                    # estimated_torque = model(target)
                    tau_ext = target

                optimizer.zero_grad()

                estimated_collision = collision_discriminator(tau_ext)

                loss = F.binary_cross_entropy_with_logits(estimated_collision, collision)

                loss.backward()
                optimizer.step()

                predicted = torch.round(torch.sigmoid(estimated_collision))
                correct = (predicted == collision).float().sum()
                accuracy = correct / collision.numel()
                
                with torch.no_grad():
                    train_loss += loss.item()*target.size(0)
                    train_accuracy += accuracy.item()*target.size(0)
                    train_total += target.size(0)
                    
            train_mean_acc = train_accuracy / train_total
            train_mean_loss = train_loss / train_total
            writer.add_scalar('train/loss', train_mean_loss, global_step=epoch)
            
        # Validation Code
        with tqdm(valid_loader, desc='VALID', position=2, leave=False, dynamic_ncols=True) as valid_bar, torch.no_grad():
            valid_loss, valid_total, valid_mean_loss = 0, 0, 0

            for batch_idx, (input, target, collision) in enumerate(valid_bar):

                _, target, collision = _, target.to(device), collision.to(device)

                # estimated_torque = model(input)
                tau_ext = target
                estimated_collision = collision_discriminator(tau_ext)
                loss = F.binary_cross_entropy_with_logits(estimated_collision, collision)

                valid_loss += loss.item()*target.size(0)
                valid_total += target.size(0)

                predicted = torch.round(torch.sigmoid(estimated_collision))
                correct = (predicted == collision).float().sum()
                accuracy = correct / collision.numel()

            valid_mean_loss = valid_loss / valid_total
            writer.add_scalar('valid/loss', valid_mean_loss, global_step=epoch)

        lr_list.append(optimizer.param_groups[0]['lr'])
        loss_list.append(train_mean_loss)
        acc_list.append(train_mean_acc)

        
        torch.save({
            'cfg': CFG,
            'last_epoch': epoch,
            'lr_list': lr_list,
            'loss_list': loss_list,
            'last_state_dict': clone_state_dict(collision_discriminator),
            'saved_hidden_size': CFG.hidden_size,
            'saved_num_layers': CFG.num_layers,
            'saved_epoch': CFG.epoch,
            'saved_learning_rate': CFG.learning_rate,
            'saved_batch_size': CFG.batch_size,
            'accuracy': acc_list, 
        }, CKPT_FILENAME)
 
        writer.add_scalar('epoch/epoch', epoch, global_step=epoch)
        writer.add_scalar('epoch/learning_rate', lr_list[-1], global_step=epoch)
                