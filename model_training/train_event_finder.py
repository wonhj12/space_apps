import os
import pandas as pd
import numpy as np
from obspy import read
from sklearn.model_selection import train_test_split

# 시계열 데이터 준비 함수 (여러 파일 처리, 라벨 데이터셋 사용)
def load_and_preprocess_data_slicing(data_dir, label_csv_path, time_step=6000, overlap_step=100):
    # 카탈로그 라벨 데이터셋 로드
    label_df = pd.read_csv(label_csv_path)
    
    # mseed 파일 목록 생성
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.mseed') and ('padded' in f)]
    
    # 파일을 8:2 비율로 train과 val로 나누기
    train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)
    
    # 데이터셋 로드 및 전처리 함수 정의
    def process_files(file_list, overlap_step_data):
        X, y = [], []
        for file_name in file_list:
            file_path = os.path.join(data_dir, file_name)
            stream = read(file_path)  # mseed 파일 읽기
            tr = stream[0]  # 트레이스 (Trace) 추출
            velocity = tr.data  # mseed 파일의 진폭 데이터를 velocity로 사용
            
            # time_rel 추출 (초 단위로 변환)
            sampling_rate = tr.stats.sampling_rate
            time_rel = np.arange(0, len(velocity)) / sampling_rate
            
            # 데이터 정규화 (옵션)
            velocity = (velocity - np.min(velocity)) / (np.max(velocity) - np.min(velocity))

            # 라벨 데이터에서 해당 파일의 time_rel 값을 가져오기
            label_row = label_df[label_df['filename'] + '_padded.mseed' == file_name]
            if not label_row.empty:
                event_time_rel = label_row['time_rel(sec)'].values[0]  # 라벨 (지진파 시작점)
                
                # event_time_rel에 해당하는 인덱스 찾기
                event_index = int(event_time_rel * sampling_rate)

                # 슬라이싱 기준을 event_index에 맞춰서 샘플링
                # start_index 범위 조정: event_index - (time_step // 2)에서 event_index까지 슬라이싱
                for start_index in range(max(0, event_index - time_step // 2), min(len(velocity) - time_step + 1, event_index + 1), overlap_step_data):
                    end_index = start_index + time_step
                    
                    # 슬라이싱이 가능한 경우에만 추가
                    if end_index <= len(velocity):
                        X.append(velocity[start_index:end_index])
                        
                        # event_time_rel을 현재 샘플 내에서의 상대적인 시간으로 변경
                        relative_event_time = event_time_rel - (start_index / sampling_rate)
                        y.append(relative_event_time)
        
        return np.array(X), np.array(y)
    
    # train과 validation 데이터셋 처리
    X_train, y_train = process_files(train_files, overlap_step)
    X_val, y_val = process_files(val_files, overlap_step_data=1000)
    
    return X_train, y_train, X_val, y_val



# 데이터가 저장된 디렉토리 경로
data_dir = './seismic_detection/data_copy/lunar/training/data/S12_GradeA/'
label_csv_path = './seismic_detection/data_copy/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv'

test_data_dir = './seismic_detection/data_copy/lunar/training/testset/'



X_train, y_train, X_val, y_val = load_and_preprocess_data_slicing(data_dir, label_csv_path, overlap_step=50)



print(len(X_train))
print(len(X_train[0]))
print(len(X_val))
print(len(y_val))



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from obspy import read
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_absolute_error

# GPU 사용 여부 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class SeismicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        return sample, label

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
        self.fc2 = nn.Linear(128, 1)  # 출력층: 회귀 값 1개

    def calculate_conv_output_size(self, input_channels):
        size = input_channels
        size = (size - 2) // 2  # conv1 and pool1
        size = (size - 2) // 2  # conv2 and pool2
        size = (size - 2) // 2  # conv3 and pool3
        return size * 256

    def forward(self, x):
        # x의 크기 변환 [batch_size, time_step] -> [batch_size, 1, time_step]
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

# 데이터셋 및 데이터로더 생성
train_dataset = SeismicDataset(X_train, y_train)
val_dataset = SeismicDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

# 모델, 손실 함수, 옵티마이저 정의
num_epochs = 100
input_size = X_train.shape[1]
model = OneDCNNRegressionModel(input_channels=input_size).to(device)  # 모델을 GPU로 이동
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-5)

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in tqdm(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    best_val = float('inf')
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            print(outputs)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item() * inputs.size(0)

            # Calculate MAE
            mae = mean_absolute_error(targets.cpu().numpy(), outputs.cpu().squeeze().numpy())
            val_mae += mae * inputs.size(0)

    val_loss /= len(val_loader.dataset)
    val_mae /= len(val_loader.dataset)
    
    if val_loss < best_val:
        torch.save(model.state_dict(), 'event_time_finder_1DCNN_2.pth')
        best_val = val_loss



    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')

import os
import pandas as pd
import numpy as np
from obspy import read

# 시계열 데이터 준비 함수 (여러 파일 처리, 라벨 데이터셋 사용)
def load_and_preprocess_data_with_event_cut(data_dir, label_csv_path, time_step=6000):
    # 카탈로그 라벨 데이터셋 로드
    label_df = pd.read_csv(label_csv_path)
    
    # mseed 파일 목록 생성
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.mseed') and ('padded' in f)]
    
    # 데이터셋 로드 및 전처리 함수 정의
    def process_files(file_list):
        X, y = [], []
        for file_name in file_list:
            file_path = os.path.join(data_dir, file_name)
            stream = read(file_path)  # mseed 파일 읽기
            tr = stream[0]  # 트레이스 (Trace) 추출
            velocity = tr.data  # mseed 파일의 진폭 데이터를 velocity로 사용
            
            # 데이터 정규화 (옵션)
            velocity = (velocity - np.min(velocity)) / (np.max(velocity) - np.min(velocity))
            
            # 라벨 데이터에서 해당 파일의 이벤트 시간 가져오기
            label_row = label_df[label_df['filename'] + '_padded.mseed' == file_name]
            if not label_row.empty:
                event_time_rel = label_row['time_rel(sec)'].values[0]  # 라벨 (지진파 시작점)
                
                # 이벤트 포인트를 중심으로 자르기
                sampling_rate = tr.stats.sampling_rate
                event_index = int(event_time_rel * sampling_rate)
                start_index = max(0, event_index - time_step // 2)
                end_index = min(len(velocity), start_index + time_step)
                
                # 슬라이싱된 데이터 추가
                X.append(velocity[start_index:end_index])
                print(event_time_rel, start_index, sampling_rate)
                relative_event_time = event_time_rel - (start_index / sampling_rate)
                y.append(relative_event_time)
        
        return np.array(X), np.array(y)
    
    # 테스트 데이터셋 처리
    X_test, y_test = process_files(file_list)
    
    return X_test, y_test

# 사용 예시
test_data_dir = './seismic_detection/data_copy/lunar/training/testset/'
label_csv_path = './seismic_detection/data_copy/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv'
X_test, y_test = load_and_preprocess_data_with_event_cut(test_data_dir, label_csv_path)
print("Test dataset shape:", X_test.shape)
print("Labels shape:", y_test.shape)

import os
import pandas as pd
import numpy as np
from obspy import read
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
import tensorflow as tf


class SeismicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        return sample, label

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
        self.fc2 = nn.Linear(128, 1)  # 출력층: 회귀 값 1개

    def calculate_conv_output_size(self, input_channels):
        size = input_channels
        size = (size - 2) // 2  # conv1 and pool1
        size = (size - 2) // 2  # conv2 and pool2
        size = (size - 2) // 2  # conv3 and pool3
        return size * 256

    def forward(self, x):
        # x의 크기 변환 [batch_size, time_step] -> [batch_size, 1, time_step]
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

def load_model(model_path, model_class, device):
    # 모델 클래스 정의된 객체를 생성한 후에 state_dict를 로드합니다.
    model = model_class(input_channels=6000)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model_path = 'event_time_finder_1DCNN.pth'  # 저장된 모델 파일 경로

# 모델 클래스 정의 필요 (예: OneDCNNClassificationModel)
model = load_model(model_path, OneDCNNRegressionModel, device)

# 데이터셋 및 데이터로더 생성
val_dataset = SeismicDataset(X_test, y_test)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

model.eval()
with torch.no_grad():
    for i, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        
        # 출력 결과 및 정답 값 출력 (첫 3개의 배치만)
        if i < 3:
            for j in range(len(inputs)):
                print(f"Sample {i * len(inputs) + j + 1} - Predicted: {outputs[j].item()}, Actual: {targets[j].item()}")

# 텐서플로우 모델 로드
tensorflow_model = tf.keras.models.load_model("event_time_finder.h5")

# 모델 평가 코드 (테스트 데이터에 대해 각각의 정답 값과 예측 값을 출력)
for i, (inputs, targets) in enumerate(val_loader):
    inputs_np = inputs.numpy()  # 텐서플로우 모델에 입력하기 위해 넘파이 배열로 변환
    predictions = tensorflow_model.predict(inputs_np)
    
    # 출력 결과 및 정답 값 출력 (첫 3개의 배치만)
    if i < 3:
        for j in range(len(inputs)):
            print(f"Sample {i * len(inputs) + j + 1} - Predicted: {predictions[j][0]}, Actual: {targets[j].item()}")
