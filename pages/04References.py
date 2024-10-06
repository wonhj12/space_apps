import streamlit as st
from PIL import Image

def display_sidebar_toc():
    st.markdown("""
    <style>
    /* Style the sidebar */
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
        padding: 20px;
    }
    
    /* Style TOC links in the sidebar */
    .toc-link {
        font-size: 20px;
        text-decoration: none;
        padding: 3px 0;
        display: block;
    }

    .toc-link:hover {
        color: #8b5543;
        font-weight: bold;
        text-decoration: underline;
    }

    </style>
    """, unsafe_allow_html=True)
    
def display_reference():
    #reference
    st.markdown("""
    <div>
        <h1 style="color : #d38856">
            Reference
        </h1>
    </div>""", unsafe_allow_html=True)
    
    # 출처를 기재하는 코드
    st.markdown("""
    <div>
        <h2 style="color : #d38856">
            Image Sources
        </h2>
    </div>""", unsafe_allow_html=True)
    st.markdown("1. Astronaut in moon : [Nvida](https://blogs.nvidia.co.kr/blog/nasa_deeplearning/)")
    st.markdown("2. Seismic Exploration on Mars : [Space Apps Challenge](https://www.spaceappschallenge.org/nasa-space-apps-2024/challenges/seismic-detection-across-the-solar-system/)")
    st.markdown("3. Seismic Exploration on the Moon : [달에서도 지구처럼 지각변동으로 지진 진행 중](https://www.yna.co.kr/view/AKR20190513141751009)")

    # 텍스트 출처
    st.markdown("""
    <div>
        <h2 style="color : #d38856">
            Text Reference
        </h2>
    </div>""", unsafe_allow_html=True)
    
    st.markdown("1. [Seismic Detection Across the Solar System](https://www.spaceappschallenge.org/nasa-space-apps-2024/challenges/seismic-detection-across-the-solar-system/)")
    st.markdown("2. [NASA’s InSight Detects First Likely ‘Quake’ on Mars](https://science.nasa.gov/missions/insight/nasas-insight-detects-first-likely-quake-on-mars/)")
    st.markdown("3. [NASA’s InSight Reveals the Deep Interior of Mars](https://www.jpl.nasa.gov/news/nasas-insight-reveals-the-deep-interior-of-mars/)")
    st.markdown("4. [Mars Seismic Deployment Lays Groundwork for Future Planetary Missions](https://www.seismosoc.org/news/mars-seismic-deployment-lays-groundwork-for-future-planetary-missions/)")
    st.markdown("5. [InSight Lander](https://science.nasa.gov/mission/insight/)")
    st.markdown("6. [달에서도 지구처럼 지각변동으로 지진 진행 중](https://www.yna.co.kr/view/AKR20190513141751009)")
    st.markdown("""
        <div>
            <h2 style="color : #d38856">
                AI Reference
            </h2>
        </div>""", unsafe_allow_html=True)
    st.markdown("1. [Chat GPT](https://chatgpt.com/)")
    st.markdown("2. [Gemini](https://gemini.google.com)")
    
# 페이지가 호출될 때 함수 실행
if __name__ == "__main__":
    display_sidebar_toc()
    display_reference()
