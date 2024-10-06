import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import json
from controller.prepare_data import read_mseed_data
from controller.model import predict
from streamlit_lottie import st_lottie

# Lottie 애니메이션 로딩
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
# 로딩 애니메이션 불러오기
lottie_animation = load_lottiefile('./animations/NASA_Loading.json')

st.set_page_config(page_title="Guguduck", layout="wide")

option = st.sidebar.selectbox(
    'Menu',
     ('페이지1', '페이지2', '페이지3'))
with st.sidebar:
    choice = option_menu("Menu", ["페이지1", "페이지2", "페이지3"],
                         icons=['house', 'kanban', 'bi bi-robot'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "4!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "—hover-color": "#fafafa"},
        "nav-link-selected": {"background-color": "#08c7b4"},
    }
    )

st.title('Seismic Detection')

# mseed 파일 업로드
uploaded_file = st.file_uploader('Upload Seismic File', label_visibility = 'hidden', type=['mseed'], accept_multiple_files=False)

if (uploaded_file is not None):
    # 로딩 애니메이션 표시
    # 중앙 정렬
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        animation_placeholder = st.empty()
        with animation_placeholder:
            st_lottie(lottie_animation, speed=1, reverse=False, loop=True, height=500, width=500)

    # 데이터 전처리
    # result, velocity, times = load_and_preprocess_data_slicing(uploaded_file)
    velocity, times, sampling_rate = read_mseed_data(uploaded_file)

    # 데이터 예측
    # event_times, 
    # , a, event_times
    classification = predict(velocity, times, sampling_rate)

    # 시각화
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(times, velocity)

    # 모델이 예측한 범위 표시
    for start, end, max_velocity in classification:
        ax.axvspan(start, end, color='yellow', alpha=0.3, label='Predicted Event Segment')

    # 이벤트 표시
    # for event in event_times:
    #     ax.axvline(x=event, color='red', linestyle='-', label='Rel. Arrival')

    ax.set_xlim([min(times),max(times)])
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Time (s)')

    st.pyplot(plt)

    # 로딩 완료 후 로딩 애니메이션 제거
    animation_placeholder.empty()
