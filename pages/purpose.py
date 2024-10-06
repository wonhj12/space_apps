import streamlit as st

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-color: #1f1f1f;
             background-attachment: fixed;
             background-size: cover
             
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

st.title("Challenge Purpose")

st.markdown("### **Background**")
# st.image("https://d2pn8kiwq2w21t.cloudfront.net/images/missionswebPIA22743-16_rfbG1OZ.2e16d0ba.fill-548x400-c50.jpg",width=300)

st.markdown("""
    <div >
        <h4 style="padding: 10px; background-color:#8b5543; border-radius: 10px;">
            Importance of Collecting Seismic Waves from Celestial Bodies
        </h4>
        <p >Seismic waves travel at different speeds and shapes as they pass through various materials within a planet. This provides a method to study <b>the internal structure</b> of the planet using seismic waves.</p>
        <p>Apollo, Viking, 그리고 InSight에서 얻은 행성 지진계 설계, 배치 및 작동, 지진 신호 처리 및 신호 해석에 대한 교훈은 우리가 이러한 지구체에 대한 최상의 지진 모니터링을 수행하고 향후 임무를 통해 태양계에 대한 더 나은 과학적 이해로 이어질 것입니다.</p>
        <h5>화성에서의 지진 탐사</h5>
        
    </div>
    <div style="display: flex; align-items: flex-start;">
        <div style="flex: 1;">
            <img src="https://d2pn8kiwq2w21t.cloudfront.net/images/missionswebPIA22743-16_rfbG1OZ.2e16d0ba.fill-548x400-c50.jpg" alt="Seismic Exploration on Mars" style="width: 400px; margin-right: 20px;"/>
        </div>
        <div style="flex: 2;">
            <p>2019년 작업을 시작한 SEIS는 화성의 지진을 탐사함. 2020년까지 400건 이상의 지진이 감지되었음.</p>
            <p>Particularly in the case of <b>Mars</b>, it provides insights into how all rocky planets, including Earth, were formed.</p>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div>
        <h4>Problem</h4>
        <p style="font-size=18px">High Cost</p>
        <p>1. Transmitting the vast amounts of data collected by spacecraft from celestial bodies back to Earth is very costly, as it requires a significant amount of power to send the data, which is greatly affected by distance.</p>
        <p>2. Only some of this data is scientifically useful.</p>
        
    </div>
""", unsafe_allow_html=True)
st.markdown("""
    <div>
            <h4>Effect</h4>
            <p>이 프로젝트가 어디에 이용되면 좋을지?</p>
            <p>1. If the event point is predicted by the probe, a threshold is set to define a range that includes the seismic waves before and after the event.</p>
            <p>2. event point가 있는 특정 구간의 지진파 정보만을 지구로 보내면 전력량을 줄일 수 있다</p>
            
    </div>
""", unsafe_allow_html=True)
