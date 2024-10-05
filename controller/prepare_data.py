import pandas as pd
import numpy as np
from obspy import read

# 시계열 데이터 준비 함수 (여러 파일 처리, 라벨 데이터셋 사용)
def load_and_preprocess_data_slicing(file):    
    time_step = 6000
    x = []

    try:
        stream = read(file)  # mseed 파일 읽기
        tr = stream[0]  # 트레이스 (Trace) 추출
        velocity = tr.data  # mseed 파일의 진폭 데이터를 velocity로 사용
        times = tr.times()

        # 데이터 정규화 (옵션)
        velocity = (velocity - np.min(velocity)) / (np.max(velocity) - np.min(velocity))

        # 데이터 slicing
        x = [velocity[i:i + time_step] for i in range(0, len(velocity), time_step)]

        if len(x[-1]) < time_step:
            x[-1] = np.pad(x[-1], (0, time_step - len(x[-1])), 'constant')
            
    except Exception as e:
        return None
    
    return np.array(x), velocity, times