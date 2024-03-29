import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

"""
Dataloader for CNN based models

input tensor dimension : [batch_size, 3, num_joints, max_seq_len]
target tensor dimension : [batch_size, 1, num_joints, max_seq_len]

"""
 
class RobrosCNN(Dataset):
    def __init__(self, train, input_folder_path=None, target_folder_path=None, num_joints=int):
        self.train = train
        self.num_joints = num_joints
        self.input_folder_path = input_folder_path
        self.target_folder_path = target_folder_path

        self.inputs = self.load_inputs()
        self.targets = self.load_targets()

        input_max_seq_len = max(max(df.shape[1] for df in joint_data) for joint_data in self.inputs.values())
        target_max_seq_len = max(df.shape[1] for df in self.targets)
        self.max_seq_len = max(input_max_seq_len, target_max_seq_len)
        # self.max_seq_len = max(max([df.shape[1] for df in joint_data]) for joint_data in self.inputs.values() + self.targets)
 
    def load_csvs_from_folder(self, folder_path):
        all_csvs = sorted(os.listdir(folder_path))
        data = [pd.read_csv(os.path.join(folder_path, csv_file), header=None) for csv_file in all_csvs]
        return data
 
    def load_inputs(self):
        inputs = {}
        for data_type in ['joint_position', 'joint_velocity', 'joint_acceleration']:
            folder_path = os.path.join(self.input_folder_path, data_type)
            inputs[data_type] = self.load_csvs_from_folder(folder_path)
        return inputs
 
    def load_targets(self):
        return self.load_csvs_from_folder(self.target_folder_path)
    
    def pad_sequence(self, sequence, max_len):
        padded = np.zeros(max_len)
        padded[:len(sequence)] = sequence
        return padded
 
    def __len__(self):
        return len(self.targets[0])

    def get_max_seq_len(self):
        return self.max_seq_len
 
    def __getitem__(self, idx):
        inputs = []
        # Load and pad input data
        for data_type in ['joint_position', 'joint_velocity', 'joint_acceleration']:
            for joint_idx in range(self.num_joints):
                data = self.inputs[data_type][joint_idx]
                seq = data.iloc[idx].dropna().values
                padded_seq = self.pad_sequence(seq, self.max_seq_len)
                inputs.append(padded_seq)
        inputs = np.array(inputs).reshape(3, self.num_joints, self.max_seq_len)
 
        targets = []
        # Load and pad target data
        for joint_idx in range(self.num_joints):
            data = self.targets[joint_idx]
            seq = data.iloc[idx].dropna().values
            padded_seq = self.pad_sequence(seq, self.max_seq_len)
            targets.append(padded_seq)
        targets = np.array(targets).reshape(1, self.num_joints, self.max_seq_len)
 
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
 
if __name__=="__main__":
    input_folder_path = '/home/rtlink/robros/dataset/robros_dataset/input_data'
    target_folder_path = '/home/rtlink/robros/dataset/robros_dataset/target_data'
    num_joints = 7
 
    train_dataset = RobrosCNN(train=True, input_folder_path=input_folder_path, target_folder_path=target_folder_path, num_joints=num_joints)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
 
    for inputs, targets in train_loader:
        print("Input Tensor Shape:", inputs.shape)   # [batch_size, 3, num_joints, max_seq_len]
        print("Target Tensor Shape:", targets.shape) # [batch_size, 1, num_joints, max_seq_len]
        break
 
    first_batch_inputs, first_batch_targets = next(iter(train_loader))
 
    first_data_input = first_batch_inputs[0]
    first_data_target = first_batch_targets[0]
 
    print("First Data Input Shape:\n", first_data_input.shape)
    print("\nFirst Data Target Shape:\n", first_data_target.shape)