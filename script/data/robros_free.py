import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import pickle
import time
 

"""
Dataloader for RNN based models
- free motion
- Seq_len

input tensor dimension : [batch_size, 3 * num_joints, seq_len]
target tensor dimension : [batch_size, num_joints, seq_len]

"""

class RobrosRNN(Dataset):
    def __init__(self, train, input_folder, target_folder, num_joints, seq_len, offset):
        self.train = train
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.num_joints = num_joints
        self.seq_len = seq_len
        self.offset = offset
        self.index = 0
        self.pickle_path = os.path.join(self.input_folder, f"{'train' if train else 'test'}_data.pkl")

        self.input_zero_tensor = torch.zeros((self.num_joints*3, self.seq_len), dtype=torch.float32)
        self.output_zero_tensor = torch.zeros((self.num_joints, self.seq_len), dtype=torch.float32)
        
        if os.path.exists(self.pickle_path):
            print("Loading data from pickle.")
            self.data = self.load_data_pickle()
        else:
            print("Loading data from CSV and creating pickle.")
            self.data = self.load_data()
            self.save_data_pickle()

    def load_data_pickle(self):
        with open(self.pickle_path, 'rb') as file:
            data = pickle.load(file)
        return data

    def save_data_pickle(self):
        with open(self.pickle_path, 'wb') as file:
            pickle.dump(self.data, file, protocol=pickle.HIGHEST_PROTOCOL)


    def load_data(self):
        # Load data from CSV files
        data = []
        for joint in range(1, self.num_joints + 1):
            joint_data = {
                'position': self.read_csv(os.path.join(self.input_folder, 'joint_position', f'fre_joint_{joint}.csv')),
                'velocity': self.read_csv(os.path.join(self.input_folder, 'joint_velocity', f'fre_joint_{joint}.csv')),
                'acceleration': self.read_csv(os.path.join(self.input_folder, 'joint_acceleration', f'fre_joint_{joint}.csv')),
                'target': self.read_csv(os.path.join(self.target_folder, f'fre_joint_{joint}.csv')),
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
                return np.array(cells, dtype=np.float32) 

 
    def __len__(self):
        return len(self.data[0]['position']) - 2 * self.seq_len
 

    def __getitem__(self, idx):
        # start_time = time.time()
        idx = self.index + self.offset
        inputs = torch.zeros((self.num_joints, 3, self.seq_len), dtype=torch.float32)
        targets = []
    
        for joint_idx, joint_data in enumerate(self.data):

            if idx+2*self.seq_len > len(self.data[0]['position']):
                return self.input_zero_tensor, self.output_zero_tensor

            position = joint_data['position'][idx:idx+self.seq_len]
            velocity = joint_data['velocity'][idx:idx+self.seq_len]
            acceleration = joint_data['acceleration'][idx:idx+self.seq_len]
            target = joint_data['target'][idx+self.seq_len:idx+2*self.seq_len]
    
            inputs[joint_idx, 0, :] = torch.tensor(position, dtype=torch.float32)
            inputs[joint_idx, 1, :] = torch.tensor(velocity, dtype=torch.float32)
            inputs[joint_idx, 2, :] = torch.tensor(acceleration, dtype=torch.float32)
            targets.append(target)
    
        self.index += self.seq_len + self.offset
    
        inputs = inputs.view(-1, 3*self.num_joints, self.seq_len)
    
        targets = torch.tensor(targets, dtype=torch.float32).view(-1, self.num_joints, self.seq_len)

        inputs = inputs.squeeze()
        targets = targets.squeeze()
        # end_time = time.time()  
        # print(f"Execution time: {end_time - start_time} seconds")

        return inputs, targets

    # def __getitem__(self, idx):
    #     start_time = time.time()  # 시작 시간 기록

    #     idx = self.index + self.offset
    #     inputs = torch.zeros((self.num_joints, 3, self.seq_len), dtype=torch.float32)
    #     targets = []
        
    #     for joint_idx, joint_data in enumerate(self.data):
    #         if idx+2*self.seq_len > len(joint_data['position']):
    #             inputs = torch.zeros((self.num_joints*3, self.seq_len), dtype=torch.float32)
    #             targets = torch.zeros((self.num_joints, self.seq_len), dtype=torch.float32)
    #             end_time = time.time()  # 종료 시간 기록
    #             print(f"Execution time: {end_time - start_time} seconds")
    #             return inputs, targets

    #         position = joint_data['position'][idx:idx+self.seq_len]
    #         velocity = joint_data['velocity'][idx:idx+self.seq_len]
    #         acceleration = joint_data['acceleration'][idx:idx+self.seq_len]
    #         target = joint_data['target'][idx+self.seq_len:idx+2*self.seq_len]
        
    #         inputs[joint_idx, 0, :] = torch.tensor(position, dtype=torch.float32)
    #         inputs[joint_idx, 1, :] = torch.tensor(velocity, dtype=torch.float32)
    #         inputs[joint_idx, 2, :] = torch.tensor(acceleration, dtype=torch.float32)
    #         targets.append(target)
        
    #     self.index += self.seq_len + self.offset
        
    #     inputs = inputs.view(-1, 3*self.num_joints, self.seq_len)
    #     targets = torch.tensor(targets, dtype=torch.float32).view(-1, self.num_joints, self.seq_len)

    #     inputs = inputs.squeeze()
    #     targets = targets.squeeze()

    #     end_time = time.time()  # 종료 시간 기록
    #     print(f"Execution time: {end_time - start_time} seconds")
    #     return inputs, targets


    

if __name__=="__main__":
    input_folder_path = '/home/rtlink/robros/dataset/0216_norm/0216_free/input_data'
    target_folder_path = '/home/rtlink/robros/dataset/0216_norm/0216_free/target_data'
    num_joints = 7 
    seq_len = 100
    offset = 100

    train_dataset = RobrosRNN(train=True, input_folder=input_folder_path, target_folder=target_folder_path, num_joints=num_joints, seq_len=seq_len, offset=offset)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    for inputs, targets in train_loader:
        print("Input Tensor Shape:", inputs.shape)   # [batch_size, 3*num_joints, seq_len]
        print("Target  Tensor Shape:", targets.shape) # [batch_size, num_joints, seq_len]
        break  
 

    first_batch_inputs, first_batch_targets = next(iter(train_loader))

    first_data_input = first_batch_inputs[0]
    first_data_target = first_batch_targets[0]

    print("First Data Input Second Dimension Tensor:\n", first_data_input.size())
    print("\nFirst Data Target Second Dimension Tensor:\n", first_data_target.size())
