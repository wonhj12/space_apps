import torch.nn as nn
import torch.nn.functional as F

class OneDCNNRegressionModel(nn.Module):
    def __init__(self, input_channels):
        super(OneDCNNRegressionModel, self).__init__()
        # Conv1D 레이어 정의
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # Fully connected 레이어 정의
        conv_output_size = self.calculate_conv_output_size(input_channels)
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)

    def calculate_conv_output_size(self, input_channels):
        size = input_channels
        size = (size - 2) // 2  # conv1 and pool1
        size = (size - 2) // 2  # conv2 and pool2
        size = (size - 2) // 2  # conv3 and pool3
        return size * 256

    def forward(self, x):
        x = x.unsqueeze(1)
        
        # Conv1, Pooling
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Conv2, Pooling
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Conv3, Pooling
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        # Flatten, Fully Connected Layer, Dropout
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # 출력층 (회귀 값 예측)
        x = self.fc2(x)
        return x