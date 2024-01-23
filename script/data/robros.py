import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import TensorDataset
import torch

from utils import functions
 
class Robros(Dataset):
    def __init__(self, train=True, folder_path='/home/rtlink/robros/dataset/robros_dataset'):

        files = os.listdir(folder_path)
        class_files = {'cls': [], 'fre': []}

        for file in files:
            if 'cls' in file:
                class_files['cls'].append(file)
            elif 'fre' in file:
                class_files['fre'].append(file)
 
        fre_data = functions.load_and_combine_files(class_files['fre'], folder_path)
 
        fre_data['label'] = 1

        combined_data = pd.concat([fre_data], ignore_index=False, axis=1)
        combined_data = combined_data.drop(combined_data.index[0])
 
        signals, labels, joints = functions.prepare_dataset(combined_data)
        self.dataset = TensorDataset(signals, labels, joints)
 
        total_size = len(self.dataset)
        train_size = int(total_size * 0.7)  # 70% 훈련 데이터
        test_size = total_size - train_size
 
        self.train_dataset, self.test_dataset = random_split(self.dataset, [train_size, test_size])
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
    
    testset = Robros(train=False)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False)
    

    for signals, labels, joints in train_loader:
        print("Train Dataset - First Batch")
        print("Signals:", signals)
        print("Labels:", labels)
        print("Joints:", joints)
        break  
    

    for signals, labels, joints in test_loader:
        print("\nTest Dataset - First Batch")
        print("Signals:", signals)
        print("Labels:", labels)
        print("Joints:", joints)
        break  
    