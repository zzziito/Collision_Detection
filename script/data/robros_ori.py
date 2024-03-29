import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from torch.utils.data import TensorDataset
import torch


import utils.functions as fn

class Robros(Dataset):
    def __init__(self, train=True, input_folder_path=None, target_folder_path=None):

        input_files = os.listdir(input_folder_path)
        target_files = os.listdir(target_folder_path)
        
        class_files = {'input_cls': [], 'input_fre': [], 'target_cls': [], 'target_fre': []}

        for file in input_files:
            if 'cls' in file:
                class_files['input_cls'].append(file)
            elif 'fre' in file:
                class_files['input_fre'].append(file)

        for file in target_files:
            if 'cls' in file:
                class_files['target_cls'].append(file)
            elif 'fre' in file:
                class_files['target_fre'].append(file)
 
        input_data = fn.load_and_combine_files(class_files['input_fre'], input_folder_path)
        target_data = fn.load_and_combine_files(class_files['target_fre'], target_folder_path)

        input_data['label'] = 1
        target_data['label'] = 1

        combined_data = pd.concat([input_data], ignore_index=False, axis=1)
        combined_data = combined_data.drop(combined_data.index[0])
        
        target_combined_data = pd.concat([target_data], ignore_index=False, axis=1)
        target_combined_data = target_combined_data.drop(target_combined_data.index[0])

 
        signals, labels, joints = fn.prepare_dataset(combined_data)
        target_signals, target_labels, target_joints = fn.prepare_dataset(target_data)

        self.input_dataset = TensorDataset(signals, labels, joints)
        self.target_dataset = TensorDataset(target_signals, target_labels, target_joints)

        total_size = len(self.input_dataset)
        train_size = int(total_size * 0.7)  # 70% 훈련 데이터
        test_size = total_size - train_size

        train_indices = torch.arange(0, train_size)
        test_indices = torch.arange(train_size, total_size)
 
        if train:
            self.subset = torch.utils.data.Subset(self.input_dataset, train_indices)
            self.target_subset = torch.utils.data.Subset(self.target_dataset, train_indices)
        else:
            self.subset = torch.utils.data.Subset(self.input_dataset, test_indices)
            self.target_subset = torch.utils.data.Subset(self.target_dataset, test_indices)
 
        self.train = train

    def __len__(self):
        return len(self.subset)
 
    def __getitem__(self, idx):
        input_data = self.subset[idx]
        target_data = self.target_subset[idx]

        inputs, labels, joints = input_data
        target_inputs, target_labels, target_joints = target_data

        return (inputs, labels, joints, target_inputs, target_labels, target_joints)
 

if __name__ == '__main__':
    input_folder_path = "/home/rtlink/robros/dataset/robros_dataset/input_data"
    target_folder_path = "/home/rtlink/robros/dataset/robros_dataset/target_data"

    train_dataset = Robros(train=True, input_folder_path=input_folder_path, target_folder_path=target_folder_path)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    test_dataset = Robros(train=False, input_folder_path=input_folder_path, target_folder_path=target_folder_path)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    
    # 샘플 개수
    sample_count = 10
    
    print("Sample data from the training dataset:")
    for i in range(sample_count):
        sample = train_dataset[i]
        inputs, labels, joints, target_inputs, target_labels, target_joints = sample
        print(f"\nSample {i+1}:")
        print(f"Inputs: {inputs}, Labels: {labels}, Joints: {joints}")
        print(f"Target Inputs: {target_inputs}, Target Labels: {target_labels}, Target Joints: {target_joints}")