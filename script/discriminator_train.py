import os,sys
from pathlib import Path
from argparse import ArgumentParser
import yaml

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import numpy as np
from torch.utils.tensorboard import SummaryWriter


from models import Discriminator
from data import RobrosDisc

#Argument Parsing
parser = ArgumentParser("Disc")

parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--learning-rate', '-lr',type=float, default=1.0E-4)
parser.add_argument('--batch-size','-bs', type=int, default=128)
parser.add_argument('--tag', type=str, required=True)

CFG = parser.parse_args()

### CUDA setup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### DataLoader Setup

input_folder = '/home/rtlink/robros/dataset/collision/regularized_torque'
target_folder = '/home/rtlink/robros/dataset/collision/target'

trainset = RobrosDisc(train=True, input_folder_path=input_folder, target_folder_path=target_folder)
validset = RobrosDisc(train=False, input_folder_path=input_folder, target_folder_path=target_folder)

loader_kwargs = dict(
    batch_size = CFG.batch_size,
    drop_last = True
)

train_loader = DataLoader(trainset, shuffle=True, **loader_kwargs)
valid_loader = DataLoader(validset, shuffle=False, **loader_kwargs)
 
len_trainset = len(trainset)
len_validset = len(validset)

### Logging

LOG_DIR = Path('./disc_log/0207')
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

### Model & Optimizer

model = Discriminator()

criterion_collision = nn.BCELoss()
criterion_joint_pos = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = CFG.batch_size
num_train_batch = len(train_loader) - 1
num_valid_batch = len(valid_loader) - 1


with tqdm(range(1, CFG.epoch + 1), desc='EPOCH', position=1, leave=False, dynamic_ncols=True) as epoch_bar:
    lr_list, loss_list = [], []

    for epoch in epoch_bar:
        with tqdm(train_loader, desc='TRAIN', position=2, leave=False, dynamic_ncols=True) as train_bar:
            train_loss, train_total, train_mean_loss = 0, 0, 0
            for batch_idx, (pos, collision, joint) in enumerate(train_bar):

                pos, collision, joint = pos.to(device), collision.to(device), joint.to(device)
                optimizer.zero_grad()
                collision_pred, joint_pos_pred = model(pos)

                loss_collision = criterion_collision(collision_pred.squeeze(), collision)
                loss_joint_pos = criterion_joint_pos(joint_pos_pred, joint)
                
                total_loss = loss_collision + loss_joint_pos
                total_loss.backward()
                optimizer.step()

                train_loss += total_loss.item()

            train_mean_loss = train_loss / len_trainset
            print(f"Epoch {epoch+1}/{CFG.epoch + 1}, Loss: {train_mean_loss:.4f}")

        print("Training complete")

        with tqdm(valid_loader, desc='VALID', position=2, leave=False, dynamic_ncols=True) as valid_bar, torch.no_grad():
            valid_loss, valid_total, valid_mean_loss = 0, 0, 0

            collision_preds, collision_trues = [], []
            joint_pos_preds = [] 
            joint_pos_trues = [] 

            for batch_idx, (pos, collision, joint) in enumerate(valid_bar):

                pos, collision, joint = pos.to(device), collision.to(device), joint.to(device)
              
                collision_pred, joint_pos_pred = model(pos)

                collision_preds.extend(collision_pred.squeeze().round().cpu().numpy())
                collision_trues.extend(collision.cpu().numpy())
                joint_pos_preds.extend(joint_pos_pred.cpu().numpy())
                joint_pos_trues.extend(pos.cpu().numpy())

        accuracy = accuracy_score(collision_trues, collision_preds)
        precision = precision_score(collision_trues, collision_preds)
        recall = recall_score(collision_trues, collision_preds)
        f1 = f1_score(collision_trues, collision_preds)
        mse = mean_squared_error(np.array(joint_pos_trues), np.array(joint_pos_preds))
        rmse = np.sqrt(mse)
    
        print(f"Collision Prediction - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        print(f"Joint Position Prediction - MSE: {mse:.4f}, RMSE: {rmse:.4f}")


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
                



 