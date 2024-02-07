import torch
import torch.nn as nn
import torch.nn.functional as F
 
class Discriminator(nn.Module):
    def __init__(self, input_features=4500, num_joints=7):
        super(Discriminator, self).__init__()
        self.num_joints = num_joints
        self.input_features = input_features
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_joints * input_features, 1024) 
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)

        # Output layer for collision detection, binary classification
        self.collision_out = nn.Linear(128, 1)
        # Output layer for joint positions, regression
        self.joint_pos_out = nn.Linear(128, num_joints)  
 
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        collision_pred = torch.sigmoid(self.collision_out(x))  
        joint_pos_pred = self.joint_pos_out(x)  
        return collision_pred, joint_pos_pred
