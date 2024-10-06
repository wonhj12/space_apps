import pandas as pd
import numpy as np
from obspy import read

# # 시계열 데이터 준비 함수 (여러 파일 처리, 라벨 데이터셋 사용)
# def load_and_preprocess_data_slicing(file):    
#     time_step = 6000
#     x = []

#     try:
#         stream = read(file)  # mseed 파일 읽기
#         tr = stream[0]  # 트레이스 (Trace) 추출
#         velocity = tr.data  # mseed 파일의 진폭 데이터를 velocity로 사용
#         times = tr.times()

#         # 데이터 정규화 (옵션)
#         velocity = (velocity - np.min(velocity)) / (np.max(velocity) - np.min(velocity))

#         # 데이터 slicing
#         x = [velocity[i:i + time_step] for i in range(0, len(velocity), time_step)]

#         if len(x[-1]) < time_step:
#             x[-1] = np.pad(x[-1], (0, time_step - len(x[-1])), 'constant')
            
#     except Exception as e:
#         return None
    
#     return np.array(x), velocity, times

# # Classification을 위한 데이터 전처리
# # Slice 된 데이터, 전체 velocity 데이터, 전체 시간 데이터를 반환
# def slice_data_for_classification(file):
#     time_step = 6000
#     x1 = []

#     try:
#         stream = read(file)  # mseed 파일 읽기
#         tr = stream[0]  # 트레이스 (Trace) 추출
#         velocity = tr.data  # mseed 파일의 진폭 데이터를 velocity로 사용
#         times = tr.times()

#         # 데이터 정규화 (옵션)
#         velocity = (velocity - np.min(velocity)) / (np.max(velocity) - np.min(velocity))

#         # 데이터 slicing (0부터 time_step 단위 만큼씩)
#         x1 = [velocity[i:i + time_step] for i in range(0, len(velocity), time_step)]
#         x2 = [velocity[i:i + time_step] for i in range(3000, len(velocity), time_step)]

#         # 길이가 짧으면 0으로 패딩 입력
#         if len(x1[-1]) < time_step:
#             x1[-1] = np.pad(x1[-1], (0, time_step - len(x1[-1])), 'constant')
#         if len(x2[-1]) < time_step:
#             x2[-1] = np.pad(x2[-1], (0, time_step - len(x2[-1])), 'constant')
            
#     except Exception as e:
#         return None
    
#     return np.array(np.concatenate([x1, x2])), velocity, times

# 데이터 정규화
# velocity, times 반환
def read_mseed_data(file): 
    try:
        stream = read(file)  # mseed 파일 읽기
        tr = stream[0]  # 트레이스 (Trace) 추출
        velocity = tr.data  # mseed 파일의 진폭 데이터를 velocity로 사용
        times = tr.times()
        sampling_rate = tr.stats.sampling_rate

        # 데이터 정규화 (옵션)
        velocity = (velocity - np.min(velocity)) / (np.max(velocity) - np.min(velocity))

    except Exception as e:
        return None
    
    return velocity, times, sampling_rate