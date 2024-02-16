import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
 
class Robros(Dataset):
    def __init__(self, train, input_folder_path, target_folder_path, num_joints, seq_len, offset):
        self.train = train
        self.input_folder_path = input_folder_path
        self.target_folder_path = target_folder_path
        self.num_joints = num_joints
        self.seq_len = seq_len
        self.offset = offset
        self.index = 0  # Initial index
        self.data = self.load_data()
 
    def load_data(self):
        # Load data from CSV files
        data = []
        for joint in range(1, self.num_joints + 1):
            joint_data = {
                'position': self.read_csv(os.path.join(self.input_folder_path, 'joint_position', f'fre_joint_{joint}.csv')),
                'velocity': self.read_csv(os.path.join(self.input_folder_path, 'joint_velocity', f'fre_joint_{joint}.csv')),
                'acceleration': self.read_csv(os.path.join(self.input_folder_path, 'joint_acceleration', f'fre_joint_{joint}.csv')),
                'target': self.read_csv(os.path.join(self.target_folder_path, f'fre_joint_{joint}.csv')),
                'collision': self.read_csv(os.path.join('0215_dataset/collision_data', f'fre_joint_{joint}.csv')),
            }
            data.append(joint_data)
        # Split data into train and test
        split_index = int(0.7 * len(data[0]['position']))
        if self.train:
            for key in data[0].keys():
                for joint_data in data:
                    joint_data[key] = joint_data[key][:split_index]
        else:
            for key in data[0].keys():
                for joint_data in data:
                    joint_data[key] = joint_data[key][split_index:]
 
        return data
 
    def read_csv(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                cells = line.strip().split(',')
                return np.array(cells, dtype=np.float32)  # Assuming all data are float
 
    def __len__(self):
        return len(self.data[0]['position']) - 2 * self.seq_len
 
    def __getitem__(self, idx):
        idx = self.index + self.offset
        inputs, targets, collisions = [], [], []
        for joint_data in self.data:
            position = joint_data['position'][idx:idx+self.seq_len]
            velocity = joint_data['velocity'][idx:idx+self.seq_len]
            acceleration = joint_data['acceleration'][idx:idx+self.seq_len]
            target = joint_data['target'][idx+self.seq_len:idx+2*self.seq_len]
            collision = joint_data['collision'][idx+self.seq_len:idx+2*self.seq_len]
            inputs.append(np.concatenate([position, velocity, acceleration], axis=0))
            targets.append(target)
            collisions.append(collision)
 
        self.index += self.seq_len + self.offset  # Update index for the next call
 
        # Convert lists to tensors
        inputs = torch.tensor(inputs, dtype=torch.float32).transpose(0, 1)  # [3*num_joints, seq_len] -> [seq_len, 3*num_joints]
        targets = torch.tensor(targets, dtype=torch.float32).transpose(0, 1)  # [num_joints, seq_len]
        collisions = torch.tensor(collisions, dtype=torch.float32).transpose(0, 1)  # [num_joints, seq_len]
 
        return inputs, targets, collisions
 
# Example of using Robros dataset
input_folder_path = '0215_dataset/input_data'
target_folder_path = '0215_dataset/target_data'
num_joints = 7
seq_len = 100  # Example sequence length
offset = 10  # Example offset
 
train_dataset = Robros(train=True, input_folder_path=input_folder_path, target_folder_path=target_folder_path, num_joints=num_joints, seq_len=seq_len, offset=offset)
test_dataset = Robros(train=False, input_folder_path=input_folder_path, target_folder_path=target_folder_path, num_joints=num_joints, seq_len=seq_len, offset=offset)
 
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
 
# You can now iterate over train_loader or test_loader in your training or evaluation loop.