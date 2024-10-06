import streamlit as st

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
    
    st.sidebar.title("Contents")
    st.sidebar.markdown("""
    1. <a class="toc-link" href="#background"> Background </a>
    2. <a class="toc-link" href="#problem"> Problem </a>
    2. <a class="toc-link" href="#effect"> Effect </a>
    """, unsafe_allow_html=True)
display_sidebar_toc()
st.title("Challenge Purpose")
st.markdown("# Background", unsafe_allow_html=True)

# st.markdown("# **Background**")
# st.image("https://d2pn8kiwq2w21t.cloudfront.net/images/missionswebPIA22743-16_rfbG1OZ.2e16d0ba.fill-548x400-c50.jpg",width=300)
# https://i0.wp.com/eos.org/wp-content/uploads/2022/11/mars-insight-impact-seismic-waves.png?w=1200&ssl=1
st.markdown("""
    <div>
        <h4>seismic exploration on Mars</h4>
        <div style="display: flex; align-items: flex-start; justify-content: space-between;">
        <div style="flex: 1; margin-right: 30px;">
            <figure>
                <img src="https://d2pn8kiwq2w21t.cloudfront.net/images/missionswebPIA22743-16_rfbG1OZ.2e16d0ba.fill-548x400-c50.jpg" style="width: 400px;"/>
                <figcaption style="text-align:center; font-size:14px;color:#555;">Seismic Exploration on Mars</figcaption>
            </figure>
        </div>
        <div style="flex: 2;">
            <p style="color:grey">Particularly in the case of <b>Mars</b>, it provides insights into how all rocky planets, including Earth, were formed.</p>
            <p style="color:grey">2019년 작업을 시작한 SEIS는 화성의 지진을 탐사함. 2020년까지 400건 이상의 지진이 감지되었음.</p>
            </div>
    </div>
        <h4>나사 탐사선</h4>
        <h4>
            Importance of Collecting Seismic Waves from Celestial Bodies
        </h4>
        <p>Seismic waves travel at different speeds and shapes as they pass through various materials within a planet. This provides a method to study <b>the internal structure</b> of the planet using seismic waves.</p>
        <p style="color:grey">Apollo, Viking, 그리고 InSight에서 얻은 행성 지진계 설계, 배치 및 작동, 지진 신호 처리 및 신호 해석에 대한 교훈은 우리가 이러한 지구체에 대한 최상의 지진 모니터링을 수행하고 향후 임무를 통해 태양계에 대한 더 나은 과학적 이해로 이어질 것입니다.</p>
    </div>
    
    
    <p style="font-size:18px; color:#EBEAFF"><b>지구와 다른점</b></p>
    <p>대부분의 사람들이 느끼는 지진은 지각판의 이동으로 인한 단층에서 비롯됨</p>
    <ul>
        <li>지구와 다르게 화성은 지각판이 없고, 대신 지각은 하나의 거대한 판과 같이 형성되어있음</li>
        <li>지구와 달리 화성은 지각판이 없고, 대신 지각은 하나의 거대한 판과 같습니다. 하지만 화성 지각에는 행성이 계속 식으면서 약간 수축하여 발생하는 응력으로 인해 단층 또는 암석 균열이 여전히 형성</li>
    </ul>
    
    """, unsafe_allow_html=True)
st.divider()
st.markdown("# Problem", unsafe_allow_html=True)

st.markdown("""
    <div>
        <h4>프로젝트로 해결해야할 문제들</h4>
        <p style="font-size:18px; color:#EBEAFF"><b>high cost</b></p>
        <ul>
            <li>Transmitting the vast amounts of data collected by spacecraft from celestial bodies back to Earth is very costly, as it requires a significant amount of power to send the data, which is greatly affected by distance.</li>
            <li>Only some of this data is scientifically useful.</li>
        </ul>
        <p style="font-size:18px; color:#EBEAFF"><b>efficiency of data transmission</b></p>
        <ul>
            <li>Only some of data is scientifically useful.</li>
        </ul>
    </div>
            
""", unsafe_allow_html=True)
st.image("images/part.png", width=500)
st.markdown("""
    <div>
        <p style="font-size:18px; color:#EBEAFF"><b>signal delay</b></p>
        <ul>
            <li>지구와 천체 사이의 거리로 인해서 신호 지연이 발생함</li>
            <li>신호가 지연되면서 간섭을 받아 데이터 손실이나 오류가 발생할 가능성이 있음</li>
        </ul>
    </div>
""", unsafe_allow_html=True)
st.divider()
st.markdown("# Effect", unsafe_allow_html=True)

st.markdown("""
    <div>
        <h4>Where can this project be applied?</h4>
        <ul>
            <li>If the event point is predicted by the probe, a threshold is set to define a range that includes the seismic waves before and after the event.</li>
            <li>event point가 있는 특정 구간의 지진파 정보만을 지구로 보내면 전력량을 줄일 수 있음</li>
            <li>효율적인 데이터만 보내면 중요한 데이터의 손실 가능성을 줄이고 데이터 신뢰도를 높일 수 있음</li>
        </ul>
        <p style="font-size:18px; color:#EBEAFF"><b>지진파 탐사로 얻을 수 있는 점</b></p>
        <div style="display: flex; align-items: flex-start; justify-content: space-between;">
        <div style="flex: 1; margin-right: 30px;">
            <figure>
                <img src="https://i0.wp.com/eos.org/wp-content/uploads/2022/11/mars-insight-impact-seismic-waves.png?w=1200&ssl=1" style="width: 400px;"/>
                <figcaption style="text-align:center; font-size:14px;color:#555;">Mars Seismic</figcaption>
            </figure>
        </div>
        <div style="flex: 2;">
            
    <p><b>1. 천체 내부 구조 이해</b></p>
    <ul>
        <li>천체 내부를 통과하면서 변하는 지진파를 분석해서 천체 내부 구조를 이해할 수 있음</li>
        <li>지진파의 굴절, 반사, 속도 감소등을 분석</li>
    </ul>
        </div>
    </div>
    <p><b>2. 탐사의 기초 데이터 제공</b></p>
    <ul>
        <li>화성의 지진파 데이터를 수집하여 향후 화성 탐사 임무의 기초 데이터를 제공할 수 있음</li>
        <li>화성의 지진파 데이터는 향후 탐사 로봇 또는 안정적인 구조물을 건설할 때, 지각에 대한 충분한 이해를 바탕으로 할 수 있도록 해줌</li>
    </ul>
    <p><b>3. 행성의 진화와 형성 과정 연구</b></p>
    <ul>
        <li>여러 천체들의 지진파 데이터를 비교해서 각 천체들의 진화와 형성 과정을 이해</li>
        <li>각 천체들의 과거 지질 활동을 알 수 있고, 현재 지질 활동의 유무를 파악</li>
        <li>암석 구조와 내부 열의 흐름을 파악</li>
    </ul>
    </div>
""", unsafe_allow_html=True)
