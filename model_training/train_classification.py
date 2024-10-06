import os
import pandas as pd
import numpy as np
from obspy import read
import random

# 시계열 데이터 준비 함수 (여러 파일 처리, 라벨 데이터셋 사용)
def load_and_preprocess_data_balanced(data_dirs, label_csv_paths, time_step=1000, overlap_step=100):
    # data_dirs: 데이터가 저장된 여러 폴더들의 경로 리스트
    # label_csv_paths: 각 폴더에 해당하는 라벨 CSV 파일들의 경로 리스트
    
    if len(data_dirs) != len(label_csv_paths):
        raise ValueError("data_dirs와 label_csv_paths의 길이가 일치해야 합니다.")
    
    all_files = []
    for data_dir, label_csv_path in zip(data_dirs, label_csv_paths):
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.mseed') and ('padded.mseed' in file_name) and 'lunar' in data_dir:
                all_files.append((data_dir, label_csv_path, file_name))
            elif file_name.endswith('.mseed') and ('_padded_randomized.mseed' in file_name) and 'extra' in data_dir:
                all_files.append((data_dir, label_csv_path, file_name))
    
    # 파일 목록을 무작위로 섞기
    random.shuffle(all_files)
    
    # 80%는 훈련, 20%는 검증으로 분할
    split_index = int(len(all_files) * 0.8)
    train_files = all_files[:split_index]
    val_files = all_files[split_index:]
    
    # 훈련 데이터 준비
    X_train, y_train = process_files(train_files, time_step, overlap_step)
    
    # 검증 데이터 준비
    X_val, y_val = process_validation_files(val_files, segment_length=6000)
    
    return (np.array(X_train), np.array(y_train)), (np.array(X_val), np.array(y_val))


def process_files(file_list, time_step, overlap_step):
    X, y = [], []

    # './false_negative_far_or_no_predicted.txt' 파일을 읽어 건너뛸 파일 리스트 생성
    with open('./false_negative_far_or_no_predicted.txt', 'r') as f:
        skip_files = set(line.strip() for line in f)

    print(len(file_list))
    for data_dir, label_csv_path, file_name in file_list:
        # 건너뛸 파일인지 확인
        if file_name in skip_files:
            continue

        # 카탈로그 라벨 데이터셋 로드
        label_df = pd.read_csv(label_csv_path)

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
        if data_dir == './seismic_detection/extra_data_copy/1h30/HHZ/40min/':
            label_row = label_df[label_df['filename'] == file_name.replace('_randomized.mseed', '')]
        else:
            label_row = label_df[label_df['filename'] == file_name.replace('_padded.mseed', '')]
        event_times = label_row['time_rel(sec)'].values if not label_row.empty else []

        count = 0
        # 이벤트 구간 우선 슬라이싱 (라벨 1)
        for event_time in event_times:
            event_index = int(event_time * sampling_rate)
            # 이벤트 주변 구간을 중심으로 슬라이싱
            for start_index in range(max(0, event_index - time_step + time_step // 4), min(len(velocity) - time_step + 1, event_index + 1), overlap_step):
                end_index = start_index + time_step
                if end_index <= len(velocity):
                    count += 1
                    X.append(velocity[start_index:end_index])
                    y.append(1)  # 이벤트가 포함된 구간이므로 라벨 1

        # 비이벤트 구간 슬라이싱 (라벨 0)
        segment_count = 0
        for start_second in range(0, len(velocity), time_step // 2):
            start_index = start_second
            end_index = start_index + time_step

            if end_index <= len(velocity):
                # 해당 구간에 이벤트가 포함되지 않은 경우에만 추가
                if not any(start_index <= event_time * sampling_rate < end_index for event_time in event_times):
                    X.append(velocity[start_index:end_index])
                    y.append(0)
                    segment_count += 1

                    # 라벨 0인 데이터의 개수를 라벨 1과 비슷하게 맞추기 위해 제한
                    if segment_count >= count:
                        break

    return X, y

def process_validation_files(file_list, segment_length=6000):
    X, y = [], []
    print(len(file_list))
    for data_dir, label_csv_path, file_name in file_list:
        if 'extra' in data_dir:
            continue
        # 카탈로그 라벨 데이터셋 로드
        label_df = pd.read_csv(label_csv_path)
        
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
        if data_dir == './seismic_detection/extra_data_copy/1h30/HHZ/40min/':
            label_row = label_df[label_df['filename'] == file_name.replace('_randomized.mseed', '')]
        else:
            label_row = label_df[label_df['filename'] == file_name.replace('_padded.mseed', '')]
        event_times = label_row['time_rel(sec)'].values if not label_row.empty else []
        
        # 6000개씩 데이터 분할
        num_segments = int(np.ceil(len(velocity) / segment_length))
        for i in range(num_segments):
            start_index = i * segment_length
            end_index = start_index + segment_length
            segment = velocity[start_index:end_index]
            
            # 끝부분 패딩 (0으로)
            if len(segment) < segment_length:
                continue
                #segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
            
            # 라벨 설정: 해당 구간에 이벤트가 포함되면 1, 아니면 0
            if any(start_index <= event_time * sampling_rate < end_index for event_time in event_times):
                y.append(1)
            else:
                y.append(0)
            
            X.append(segment)
    
    return X, y

# 데이터가 저장된 디렉토리 경로
train_data_dir = ['./seismic_detection/data_copy/lunar/training/data/S12_GradeA/']
train_label_csv_path = ['./seismic_detection/data_copy/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv']

train, val = load_and_preprocess_data_balanced(train_data_dir, train_label_csv_path, time_step=6000, overlap_step=25)

X_train, y_train = train
X_val, y_val = val

X_1 = X_train[y_train==1]
X_0 = X_train[y_train==0]
print(len(X_train))
print(len(X_1))
print(len(X_0))
#print(X[0])
print(len(X_train[0]))
print(len(X_val))
X_2 = X_val[y_val==1]
X_3 = X_val[y_val==0]
print(len(X_2))
print(len(X_3))

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

# GPU 사용 여부 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터셋 클래스 정의
class SeismicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 1D CNN 모델 정의 (이진 분류용)
class OneDCNNClassificationModel(nn.Module):
    def __init__(self, input_channels):
        super(OneDCNNClassificationModel, self).__init__()
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
        self.fc2 = nn.Linear(128, 2)  # 출력층: 클래스 0 또는 1

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
        
        # 출력층 (이진 분류)
        x = self.fc2(x)
        return x

# 데이터셋 및 데이터로더 생성
train_dataset = SeismicDataset(X_train, y_train)
val_dataset = SeismicDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

# 모델, 손실 함수, 옵티마이저 정의
input_size = X_train.shape[1]
model = OneDCNNClassificationModel(input_channels=input_size).to(device)  # 모델을 GPU로 이동
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# 모델 학습
num_epochs = 100
best_val = float('inf')
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for inputs, labels in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch"):
        # 입력값과 라벨을 GPU로 이동
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 모델에 입력값을 통과시켜 예측값 계산
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 옵티마이저 초기화 및 역전파 수행
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 손실값 및 정확도 계산
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_accuracy = correct_predictions / total_predictions
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    
    # 검증 데이터셋에 대한 평가
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    count = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            # 입력값과 라벨을 GPU로 이동
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
                
            for i in range(len(predicted)):
                if predicted[i] == 1:
                    count += 1
    val_accuracy = val_correct / val_total
    val_loss = val_loss / len(val_dataset)
    print(f"Validation Loss after Epoch [{epoch+1}/{num_epochs}]: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    print("count:", count)
    if val_loss < best_val:
        torch.save(model.state_dict(), 'original_mars_seismic_classifier.pth')
        best_val = val_loss

# 모델 평가 함수
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            # 입력값과 라벨을 GPU로 이동
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')

evaluate_model(model, val_loader)

import os
import pandas as pd
import numpy as np
from obspy import read
import random
import torch
import matplotlib.pyplot as plt

def load_model(model_path, model_class, device):
    # 모델 클래스 정의된 객체를 생성한 후에 state_dict를 로드합니다.
    model = model_class(input_channels = 6000)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_test_data(test_data_dirs, label_csv_paths, model, segment_length=6000):
    # test_data_dirs: 테스트 데이터가 저장된 폴더들의 경로 리스트
    # label_csv_paths: 각 폴더에 해당하는 라벨 CSV 파일들의 경로 리스트
    
    if len(test_data_dirs) != len(label_csv_paths):
        raise ValueError("test_data_dirs와 label_csv_paths의 길이가 일치해야 합니다.")
    
    all_files = []
    for data_dir, label_csv_path in zip(test_data_dirs, label_csv_paths):
        for file_name in os.listdir(data_dir):
            if file_name.endswith('_padded.mseed'):
                all_files.append((data_dir, label_csv_path, file_name))
    
    # 파일 목록을 무작위로 섞기
    random.shuffle(all_files)
    
    # 테스트 데이터 처리
    model.eval()  # 모델을 평가 모드로 전환
    for data_dir, label_csv_path, file_name in all_files:
        # 카탈로그 라벨 데이터셋 로드
        label_df = pd.read_csv(label_csv_path)
        
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
        label_row = label_df[label_df['filename'] == file_name.replace('_padded.mseed', '')]
        event_times = label_row['time_rel(sec)'].values if not label_row.empty else []
        
        # 전체 데이터를 한 번에 예측 및 그래프 표시
        predicted_segments = []
        num_segments = int(np.ceil(len(velocity) / segment_length))
        for i in range(num_segments):
            start_index = i * segment_length
            end_index = start_index + segment_length
            segment = velocity[start_index:end_index]
            
            # 끝부분 패딩 (0으로)
            if len(segment) < segment_length:
                continue
                #segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
            
            # 모델을 사용하여 예측값 생성
            segment = torch.tensor(segment).float().unsqueeze(0).to(device)  # 입력 형식에 맞게 변환 및 GPU로 이동
            with torch.no_grad():
                outputs = model(segment)
                predicted_label = torch.argmax(outputs, dim=1).item()
                if predicted_label == 1:
                    predicted_segments.append((start_index / sampling_rate, end_index / sampling_rate))
        
        # 그래프 생성 및 표시
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax.plot(time_rel, velocity, label='Velocity')
        
        # 이벤트 시간 표시
        for event_time in event_times:
            ax.axvline(x=event_time, color='red', linestyle='--', label='Event Time')
        
        # 모델이 예측한 범위 표시
        for start, end in predicted_segments:
            ax.axvspan(start, end, color='yellow', alpha=0.3, label='Predicted Event Segment')
        
        ax.set_xlim([min(time_rel), max(time_rel)])
        ax.set_ylabel('Velocity (m/s)')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'{file_name} - Full Data', fontweight='bold')
        ax.legend(loc='upper left')
        plt.show()

# 사용 예시
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'seismic_classifier.pth'  # 저장된 모델 파일 경로

# 모델 클래스 정의 필요 (예: OneDCNNClassificationModel)
model = load_model(model_path, OneDCNNClassificationModel, device)

test_data_dirs = ['./seismic_detection/data_copy/lunar/training/testset']
label_csv_paths = ['./seismic_detection/data_copy/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv']
evaluate_test_data(test_data_dirs, label_csv_paths, model)  # model은 미리 학습된 모델 객체


