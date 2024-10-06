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

def display_team():    
    #our-team
    st.markdown("""
    <div>
        <h1 style="color : #d38856">
            Hi, we are team Guguduck!!
        </h1>
    </div>""", unsafe_allow_html=True)
    
    # 이미지 로드
    img = Image.open("images/ALL.png")
    
    # 이미지 회전 (왼쪽으로 90도 회전)
    img = img.rotate(90, expand=True)
    
    # 회전된 이미지 출력
    st.image(img, caption="Team Guguduck")
    # 두 명의 팀원을 나란히 배치
    col1, col2 = st.columns(2)

    # 팀원 1 정보
    with col1:
        st.header("Hajin Won")
        st.image("images/원하진.png", caption="Team leader")
        st.write("""
        **Position:** Lead Scientist  
        """)

    # 팀원 2 정보
    with col2:
        st.header("Eunseo Yang")
        st.image("images/양은서.png", caption="Team Member")
        st.write("""
        **Position:** AI Engineer  
        """)
        
    # 두 명의 팀원을 나란히 배치
    col3, col4 = st.columns(2)

    # 팀원 1 정보
    with col3:
        st.header("Wonjun Lee")
        st.image("images/이원준.png", caption="Team Member")
        st.write("""
        **Position:** Software Engineer  
        """)

    # 팀원 2 정보
    with col4:
        st.header("Taekwan Kim")
        st.image("images/김태관.png", caption="Team Member")
        st.write("""
        **Position:** AI Engineer  
        """)

    # 두 명의 팀원을 나란히 배치
    col5, col6 = st.columns(2)

    # 팀원 1 정보
    with col5:
        st.header("Dayoung Choi")
        st.image("images/최다영.png", caption="Team Member")
        st.write("""
        **Position:** Software Engineer   
        """)

    # 팀원 2 정보
    with col6:
        st.header("Taewoo Kim")
        st.image("images/김태우.png", caption="Team Member")
        st.write("""
        **Position:** Data Scientist  
        """)
    
# 페이지가 호출될 때 함수 실행
if __name__ == "__main__":
    display_sidebar_toc()
    display_team()
