import streamlit as st
from controller.prepare_data import load_and_preprocess_data_slicing
from controller.model import predict
import matplotlib.pyplot as plt
import numpy as np
import random

st.set_page_config(page_title="NASA Apps", layout="wide")

# 페이지 선택
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["Purpose", "Working Process", "Our Team"])

# 선택된 페이지에 따라 내용 표시
if page == "Purpose":
    import pages.purpose
elif page == "Working Process":
    import pages.working_process
elif page == "Our Team":
    import pages.our_team

st.title('Seismic Detection')

uploaded_file = st.file_uploader('Upload Seismic File', label_visibility = 'hidden', type=['mseed'], accept_multiple_files=False)

if (uploaded_file is not None):
    # 데이터 전처리
    result, velocity, times = load_and_preprocess_data_slicing(uploaded_file)
    
    # 데이터 예측
    event_times = predict(result)

    # 랜덤으로 이벤트 선택
    num_samples = len(event_times) // 5
    random_events = random.sample(event_times.tolist(), num_samples)

    # 시각화
    fig, ax = plt.subplots(1,1,figsize=(10,3))
    ax.plot(times, velocity)

    # 이벤트 표시
    for event in random_events:
        ax.axvline(x=event, color='red', linestyle='--', label='Rel. Arrival')

    ax.set_xlim([min(times),max(times)])
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Time (s)')

    st.pyplot(plt)
