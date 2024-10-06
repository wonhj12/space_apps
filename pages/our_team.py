import streamlit as st
from PIL import Image
import pandas as pd  # pandas for DataFrame handling
import base64  # for encoding CSV file to base64

def display_team():
    # 예시 데이터프레임
    # df = pd.DataFrame({
    #     'Name': ['원하진', '양은서'],
    #     'Position': ['Lead Scientist', 'Software Engineer']
    # })

    # # CSV 파일로 변환
    # csv = df.to_csv(index=False).encode()  # DataFrame을 CSV로 변환, index는 제외
    # b64 = base64.b64encode(csv).decode()  # CSV 파일을 Base64로 인코딩

    # # 다운로드 링크 생성
    # href = f'<a href="data:file/csv;base64,{b64}" download="team_data.csv">Download CSV File</a>'
    
    # # Streamlit에 HTML 마크다운 추가
    # st.markdown(href, unsafe_allow_html=True)

    # st.markdown(href, unsafe_allow_html=True)
    
    st.title("Our Team")
    
    # 이미지 로드
    img = Image.open("images/ALL.png")
    
    # 이미지 회전 (왼쪽으로 90도 회전)
    img = img.rotate(90, expand=True)
    
    # 회전된 이미지 출력
    st.image(img, caption="Our Team")
    # 두 명의 팀원을 나란히 배치
    col1, col2 = st.columns(2)

    # 팀원 1 정보
    with col1:
        st.header("원하진")
        st.image("images/원하진.png", caption="Team leader")
        st.write("""
        **Position:** Lead Scientist  
        """)

    # 팀원 2 정보
    with col2:
        st.header("양은서")
        st.image("images/양은서.png", caption="Team Member")
        st.write("""
        **Position:** Software Engineer  
        """)
        
    # 두 명의 팀원을 나란히 배치
    col3, col4 = st.columns(2)

    # 팀원 1 정보
    with col3:
        st.header("이원준")
        st.image("images/이원준.png", caption="Team Member")
        st.write("""
        **Position:** Lead Scientist  
        """)

    # 팀원 2 정보
    with col4:
        st.header("김태관")
        st.image("images/김태관.png", caption="Team Member")
        st.write("""
        **Position:** Software Engineer  
        """)

    # 두 명의 팀원을 나란히 배치
    col5, col6 = st.columns(2)

    # 팀원 1 정보
    with col5:
        st.header("최다영")
        st.image("images/최다영.png", caption="Team Member")
        st.write("""
        **Position:** Lead Scientist   
        """)

    # 팀원 2 정보
    with col6:
        st.header("김태우")
        st.image("images/김태우.png", caption="Team Member")
        st.write("""
        **Position:** Software Engineer  
        """)
        
    st.title("Reference")
    
    # 출처를 기재하는 코드
    st.markdown("### Image Sources")
    st.markdown("1. Astronaut in moon : [Nvida](https://blogs.nvidia.co.kr/blog/nasa_deeplearning/)")
    st.markdown("2. Seismic Exploration on Mars : [Space Apps Challenge](https://www.spaceappschallenge.org/nasa-space-apps-2024/challenges/seismic-detection-across-the-solar-system/)")
    # 텍스트 출처
    st.markdown("### Text Reference")
    st.markdown("1. [Seismic Detection Across the Solar System](https://www.spaceappschallenge.org/nasa-space-apps-2024/challenges/seismic-detection-across-the-solar-system/)")
    st.markdown("2. [NASA’s InSight Detects First Likely ‘Quake’ on Mars](https://science.nasa.gov/missions/insight/nasas-insight-detects-first-likely-quake-on-mars/)")
    st.markdown("3. [NASA’s InSight Reveals the Deep Interior of Mars](https://www.jpl.nasa.gov/news/nasas-insight-reveals-the-deep-interior-of-mars/)")
    st.markdown("4. [Mars Seismic Deployment Lays Groundwork for Future Planetary Missions](https://www.seismosoc.org/news/mars-seismic-deployment-lays-groundwork-for-future-planetary-missions/)")
    
    st.markdown("### Reference AI")
    st.markdown("1. [Chat GPT](https://chatgpt.com/)")
    
# 페이지가 호출될 때 함수 실행
if __name__ == "__main__":
    display_team()
