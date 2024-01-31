import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
 
class Robros(Dataset):

    def __init__(self, train=True, input_folder_path="/home/rtlink/robros/dataset/robros_dataset/input_data", target_folder_path="/home/rtlink/robros/dataset/robros_dataset/target_data"):

        self.train = train
        self.input_files = sorted([os.path.join(input_folder_path, f) for f in os.listdir(input_folder_path) if f.startswith('fre_joint_')])
        self.target_files = sorted([os.path.join(target_folder_path, f) for f in os.listdir(target_folder_path) if f.startswith('fre_joint_')])

        self.max_seq_length = self._find_max_seq_length()

        self.num_joints = 7

        self.split_idx = int(0.7 * len(self.input_files))

        if self.train:
            self.input_files = self.input_files[:self.split_idx]
            self.target_files = self.target_files[:self.split_idx]

        else:
            self.input_files = self.input_files[self.split_idx:]
            self.target_files = self.target_files[self.split_idx:]
 
    def _find_max_seq_length(self):

        max_length = 0
        for file_path in self.input_files + self.target_files:
            df = pd.read_csv(file_path)
            if df.shape[1] > max_length:
                max_length = df.shape[1]

        return max_length
 
    def __len__(self):

        return len(self.input_files)

    def __getitem__(self, idx):

        input_data = pd.read_csv(self.input_files[idx]).fillna(0).values.flatten()
        target_data = pd.read_csv(self.target_files[idx]).fillna(0).values.flatten()

        input_pad_length = max(0, self.max_seq_length - len(input_data))
        target_pad_length = max(0, self.max_seq_length - len(target_data))

        input_padded = np.pad(input_data, (0, input_pad_length), mode='constant', constant_values=0)
        target_padded = np.pad(target_data, (0, target_pad_length), mode='constant', constant_values=0)

        inputs = torch.tensor(input_padded, dtype=torch.float)
        target_inputs = torch.tensor(target_padded, dtype=torch.float)

        joint_index = int(self.input_files[idx].split('_')[-1].split('.')[0])
        joints_one_hot = np.zeros((self.num_joints, ))
        joints_one_hot[joint_index] = 1
        joints_one_hot = torch.tensor(joints_one_hot, dtype=torch.float)

        target_joint_index = int(self.target_files[idx].split('_')[-1].split('.')[0])
        target_joints_one_hot = np.zeros((self.num_joints, ))
        target_joints_one_hot[target_joint_index] = 1
        target_joints_one_hot = torch.tensor(target_joints_one_hot, dtype=torch.float)

        return inputs, joints_one_hot, target_inputs, target_joints_one_hot
 

if __name__ == '__main__':

    input_folder_path = "/home/rtlink/robros/dataset/robros_dataset/input_data"
    target_folder_path = "/home/rtlink/robros/dataset/robros_dataset/target_data"

    train_dataset = Robros(train=True, input_folder_path=input_folder_path, target_folder_path=target_folder_path)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    test_dataset = Robros(train=False, input_folder_path=input_folder_path, target_folder_path=target_folder_path)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)


    for inputs, joints_one_hot, target_inputs, target_joints_one_hot in train_dataloader:

        print("Inputs:", inputs)
        print("Joints:", joints_one_hot)
        print("Target Inputs:", target_inputs)
        print("Target Joints:", target_joints_one_hot)

        break  # 첫 번째 배치만 출력
