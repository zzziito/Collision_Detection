import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import json 
 
class Discriminator(Dataset):
    def __init__(self, input_folder, target_folder, train=True, split_ratio=0.7):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.data = []
        self.max_seq_len = 0
        # Load data
        for file_name in os.listdir(input_folder):
            input_path = os.path.join(input_folder, file_name)
            target_path = os.path.join(target_folder, file_name)
            # Read data
            input_data = pd.read_csv(input_path, header=None).values
            target_data = pd.read_csv(target_path, header=None, converters={1: eval}).values
            # Update max_seq_len
            self.max_seq_len = max(self.max_seq_len, input_data.shape[1])
            # Append data
            for input_row, target_row in zip(input_data, target_data):
                self.data.append((input_row, target_row))
        # Split data into train and test sets
        train_size = int(len(self.data) * split_ratio)
        test_size = len(self.data) - train_size
        train_data, test_data = random_split(self.data, [train_size, test_size])
        self.data = train_data if train else test_data
 
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        
        # input_seq는 이미 적절한 형태의 numpy 배열이어야 함
        # Padding input sequence to max_seq_len
        padded_input = np.pad(input_seq, (0, self.max_seq_len - len(input_seq)), 'constant', constant_values=0)
        
        input_tensor = torch.tensor(padded_input, dtype=torch.float)
        
        label = target_seq[0]
        joints = np.array(eval(target_seq[1]))  # 문자열에서 리스트로 변환
        
        label_tensor = torch.tensor([label], dtype=torch.float)
        joints_tensor = torch.tensor(joints, dtype=torch.float)
        
        # Combine label and joints into a single target tensor
        target_tensor = torch.cat((label_tensor, joints_tensor), dim=0)
        
        return input_tensor, target_tensor




def test_dataloader_dimensions(loader):
    for input_tensor, target_tensor in loader:
        print("Input Tensor Shape:", input_tensor.shape)
        print("Target Tensor Shape:", target_tensor.shape)
        break  # Only process the first batch
 
if __name__ == "__main__":
    # Define dataset paths
    input_folder = '/home/rtlink/robros/dataset/collision/len50/cleaned/discriminator/position'
    target_folder = '/home/rtlink/robros/dataset/collision/len50/cleaned/discriminator/target'
 
    # Create the datasets
    dataset_train = Discriminator(input_folder, target_folder, train=True)
    dataset_test = Discriminator(input_folder, target_folder, train=False)
 
    # Create DataLoaders
    train_loader = DataLoader(dataset_train, batch_size=2, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=2, shuffle=False)
 
    # Test DataLoader dimensions
    print("Train DataLoader:")
    test_dataloader_dimensions(train_loader)
 
    print("\nTest DataLoader:")
    test_dataloader_dimensions(test_loader)