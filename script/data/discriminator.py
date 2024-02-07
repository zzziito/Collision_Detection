import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
 
class RobrosDisc(Dataset):
    def __init__(self, train=True, input_folder_path=None, target_folder_path=None):
        self.train = train
        self.input_folder_path = input_folder_path
        self.target_folder_path = target_folder_path
 
        # Load target data
        target_path = os.path.join(self.target_folder_path, 'target.csv')
        self.target_data = pd.read_csv(target_path)
 
        # Load input data
        self.input_data = []
        for i in range(1, 8):  # Assuming there are 7 files named fre_joint_1.csv to fre_joint_7.csv
            file_path = os.path.join(self.input_folder_path, f'fre_joint_{i}.csv')
            joint_data = pd.read_csv(file_path, header=None)
            self.input_data.append(joint_data)
 
        # Convert list of DataFrames to a single numpy array for better handling
        self.input_data = np.stack([df.values for df in self.input_data], axis=1)  # Shape: (num_samples, 7, 4500)
 
        # Split dataset
        num_samples = self.input_data.shape[0]
        split_idx = int(num_samples * 0.7)
        if self.train:
            self.input_data = self.input_data[:split_idx]
            self.target_data = self.target_data.iloc[:split_idx]
        else:
            self.input_data = self.input_data[split_idx:]
            self.target_data = self.target_data.iloc[split_idx:]
 
    def __len__(self):
        return len(self.target_data)
 
    def __getitem__(self, idx):
        reg_pos_tensor = torch.tensor(self.input_data[idx], dtype=torch.float)
        collision_tensor = torch.tensor(self.target_data.iloc[idx, 0], dtype=torch.float)
        joint_positions_tensor = torch.tensor(self.target_data.iloc[idx, 1:].values, dtype=torch.float)
        return reg_pos_tensor, collision_tensor, joint_positions_tensor

 
if __name__ == "__main__":
    # Define dataset paths
    input_folder = '/home/rtlink/robros/dataset/collision/regularized_torque'
    target_folder = '/home/rtlink/robros/dataset/collision/target'
 
    # Create the datasets
    dataset_train = RobrosDisc(train=True, input_folder_path = input_folder, target_folder_path = target_folder)
    dataset_test = RobrosDisc(train=False, input_folder_path = input_folder, target_folder_path = target_folder)
 
    # Create DataLoaders
    train_loader = DataLoader(dataset_train, batch_size=2, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=2, shuffle=False)
 
    # 훈련 데이터 로더를 이용한 차원 확인
    print("Training Data:")
    for reg_pos_tensor, collision_tensor, joint_positions_tensor in train_loader:
        print("Regularized Position Tensor Shape:", reg_pos_tensor.shape)
        print("Collision Tensor Shape:", collision_tensor.shape)
        print("Joint Positions Tensor Shape:", joint_positions_tensor.shape)
        break 