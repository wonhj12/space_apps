import streamlit as st
import matplotlib.pyplot as plt
import json
from controller.prepare_data import read_mseed_data
from controller.model import predict
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu

# Lottie 애니메이션 로딩
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
# 로딩 애니메이션 불러오기
lottie_animation = load_lottiefile('./animations/NASA_Loading.json')

# 데이터 예측 진행
def detect_seismic(file):
    if file is not None:
        # 로딩 애니메이션 표시
        # 중앙 정렬
        animation_placeholder = st.empty()
        with animation_placeholder:
            st_lottie(lottie_animation, speed=1, reverse=False, loop=True, height=500, width=500)

        # 데이터 전처리
        # result, velocity, times = load_and_preprocess_data_slicing(uploaded_file)
        velocity, times, sampling_rate = read_mseed_data(uploaded_file)

        segment_length = 6000

        # 데이터 예측
        classified_events, event_times = predict(velocity, sampling_rate, segment_length)

        # 시각화
        _, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax.plot(times, velocity)

        # 이벤트 표시
        for i in range(len(event_times)):
            if (len(classified_events) > 0):
                # Classification이 된 경우
                start, end, _ = classified_events[i]
                ax.axvspan(start, end, color='yellow', alpha=0.3, label='Predicted Event Segment')
            else:
                # Classification이 되지 않은 경우
                start = i * segment_length / sampling_rate
                if i % 2 != 0:
                    continue
            event = event_times[i]
            rel_time = event + start
            ax.axvline(x=rel_time, color='red', linestyle='-', label='Rel. Arrival')

        ax.set_xlim([min(times),max(times)])
        ax.set_ylabel('Velocity (m/s)')
        ax.set_xlabel('Time (s)')

        # 로딩 완료 후 로딩 애니메이션 제거
        animation_placeholder.empty()

        return plt

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

if 'detect_clicked' not in st.session_state:
    st.session_state.detect_clicked = False

if uploaded_file is not None and not st.session_state.detect_clicked:
    _, col, _ = st.columns([1, 1, 1])
    with col:
        if st.button('Detect', use_container_width=True):
            st.session_state.detect_clicked = True

if st.session_state.detect_clicked and uploaded_file is not None:
    st.session_state.detect_clicked = False
    graph = detect_seismic(uploaded_file)
    if graph is not None:
        st.pyplot(graph, use_container_width=True)