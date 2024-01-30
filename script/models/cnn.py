import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_channels, num_classes, sequence_length):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

        self.out = nn.Linear(256, num_classes)

        self.sequence_length = sequence_length

        self.sequence_weights = self.create_centered_weights(sequence_length)

    def create_centered_weights(self, length):
        # 시퀀스 중앙에 더 높은 가중치를 부여하는 배열 생성
        center = length // 2
        weights = torch.abs(torch.arange(length) - center)
        weights = 1 - (weights / max(weights))
        return weights

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x * self.sequence_weights

        x = F.avg_pool1d(x, kernel_size=self.sequence_length)

        # 출력
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x
