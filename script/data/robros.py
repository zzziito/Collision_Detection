import os
from torch.utils.data import Dataset, DataLoader, Sampler
import torch

class Robros(Dataset):
    def __init__(self, train: bool=True, 
                 root: str='/home/rtlink/robros/dataset/robros_dataset'):
        
        self.__split = 'train' if train else 'test'