import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import TensorDataset
import torch

import utils.functions
 
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
 
        input_data = functions.load_and_combine_files(class_files['input_fre'], input_folder_path)
        target_data = functions.load_and_combine_files(class_files['target_fre'], target_folder_path)
        
        input_data['label'] = target_data['label']

        combined_data = pd.concat([input_data], ignore_index=False, axis=1)
        combined_data = combined_data.drop(combined_data.index[0])
        
        target_combined_data = pd.concat([target_data], ignore_index=False, axis=1)
        target_combined_data = target_combined_data.drop(target_combined_data.index[0])

 
        signals, labels, joints = functions.prepare_dataset(combined_data)
        target_signals, target_labels, target_joints = functions.prepare_dataset(target_data)

        self.input_dataset = TensorDataset(signals, labels, joints)
        self.target_dataset = TensorDataset(target_signals, target_labels, target_joints)

        total_size = len(self.input_dataset)
        train_size = int(total_size * 0.7)  # 70% 훈련 데이터
        test_size = total_size - train_size
 
        self.train_dataset, self.test_dataset = random_split(self.input_dataset, [train_size, test_size])
        self.target_train_dataset, self.target_test_dataset = random_split(self.target_dataset, [train_size, test_size])

        self.train = train
 
    def __len__(self):
        if self.train:
            return len(self.train_dataset)
        else:
            return len(self.test_dataset)
 
    def __getitem__(self, idx):
        if self.train:
            return self.train_dataset[idx]
        else:
            return self.test_dataset[idx]
 

if __name__ == '__main__':

    trainset = Robros(train=True)
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    for signals, labels, joints in train_loader:
        print("Train Dataset - First Batch")
        print("Signals:", signals)
        print("Labels:", labels)
        print("Joints:", joints)
        break

    # 타겟 데이터셋 로드 및 확인
    target_trainset = Robros(train=True, input_folder_path='/path/to/target/folder', target_folder_path='/path/to/target/folder')
    target_train_loader = DataLoader(target_trainset, batch_size=64, shuffle=True)
    for signals, labels, joints in target_train_loader:
        print("\nTarget Train Dataset - First Batch")
        print("Signals:", signals)
        print("Labels:", labels)
        print("Joints:", joints)
        break