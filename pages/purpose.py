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

st.markdown("""
    <div>
        <h1 style="color : #d38856">
            Background
        </h1>
    </div>""", unsafe_allow_html=True)

# st.markdown("# **Background**")
# st.image("https://d2pn8kiwq2w21t.cloudfront.net/images/missionswebPIA22743-16_rfbG1OZ.2e16d0ba.fill-548x400-c50.jpg",width=300)
# https://i0.wp.com/eos.org/wp-content/uploads/2022/11/mars-insight-impact-seismic-waves.png?w=1200&ssl=1
st.markdown("""
    <div>
        <h2 style="color : #d38856">
            Seismic exploration on Mars
        </h2>
    </div>""", unsafe_allow_html=True)
st.markdown("""
    <div>
        <div style="display: flex; align-items: flex-start; justify-content: space-between;">
        <div style="flex: 1; margin-right: 30px;">
            <figure>
                <img src="https://d2pn8kiwq2w21t.cloudfront.net/images/missionswebPIA22743-16_rfbG1OZ.2e16d0ba.fill-548x400-c50.jpg" style="width: 400px;"/>
                <figcaption style="text-align:center; font-size:14px;color:#555;">Seismic Exploration on Mars</figcaption>
            </figure>
        </div>
        <div style="flex: 2;">
            <p style="color:grey">Particularly in the case of <b>Mars</b>, it provides insights into how all rocky planets, including Earth, were formed.</p>
            <p style="color:grey">The SEIS (Seismic Experiment for Interior Structure) project began its work in 2019 to explore seismic activity on Mars. By 2020, it had detected over 400 seismic events.</p>
        </div>
    </div>
        <ul>
            <li>InSight Lander was a Mars mission that aimed to study the planet's interior. One of the most significant challenges was the need to conserve power for long-range data transmission to Earth. The harsh space environment, filled with various types of noise, also posed a risk of data corruption, making the development of efficient algorithms for processing and transmitting seismic data essential.</li>
            <li>In order to unravel the mysteries of Mars' interior, scientists had to overcome the challenges of long-range data transmission and the harsh space environment. By developing efficient algorithms and prioritizing power conservation, the spacecraft can be able to successfully transmit seismic data back to Earth.</li>
        </ul>
        <h2 style="color : #d38856">Seismic exploration on the Moon</h2>
        <div style="display: flex; align-items: flex-start; justify-content: space-between;">
        <div style="flex: 1; margin-right: 30px;">
            <figure>
                <img src="https://img6.yna.co.kr/etc/inner/KR/2019/05/13/AKR20190513141751009_01_i_P4.jpg" style="width: 400px;"/>
                <figcaption style="text-align:center; font-size:14px;color:#555;">Seismic Exploration on the Moon</figcaption>
            </figure>
        </div>
        <div style="flex: 2;">
            <p style="color:grey">The Moon is only one-fourth the radius of Earth, so it was believed that geological activity had already ceased a long time ago.</p>
            <p style="color:grey">It has been revealed that earthquakes still occur due to the contraction of the Moon's interior.</p>
        </div>
    </div>
    <h2 style="color : #d38856">
            Importance of Collecting Seismic Waves from Celestial Bodies
        </h2>
        <p>Seismic waves travel at different speeds and shapes as they pass through various materials within a planet. This provides a method to study <b>the internal structure</b> of the planet using seismic waves.</p>
        <p style="color:grey">The lessons learned from the design, deployment, and operation of planetary seismometers from missions such as Apollo, Viking, and InSight will enhance our ability to conduct effective seismic monitoring of these celestial bodies. This knowledge is crucial for improving future missions and contributing to a better scientific understanding of the solar system.</p>
    </div>
    """, unsafe_allow_html=True)
st.divider()

st.markdown("""
    <div>
        <h1 style="color : #d38856">
            Problem
        </h1>
    </div>""", unsafe_allow_html=True)

st.markdown("""
    <div>
        <h2 style="color : #d38856">Problems to Solve with the Project</h2>
        <p style="font-size:18px; color:#EBEAFF"><h3>high cost</h3></p>
        <ul>
            <li>Transmitting the vast amounts of data collected by spacecraft from celestial bodies back to Earth is very costly, as it requires a significant amount of power to send the data, which is greatly affected by distance.</li>
        </ul>
        <p style="font-size:18px; color:#EBEAFF"><h3>efficiency of data transmission</h3></p>
        <ul>
            <li>Only some of data is scientifically useful.</li>
        </ul>
    </div>
            
""", unsafe_allow_html=True)
st.image("images/part.png", width=500)
st.markdown("""
    <div>
        <p style="font-size:18px; color:#EBEAFF"><h3>signal delay</h3></p>
        <ul>
            <li>The distance between Earth and celestial bodies causes signal delays.</li>
            <li>Signal delays can lead to interference, increasing the likelihood of data loss or errors during transmission.</li>
        </ul>
    </div>
""", unsafe_allow_html=True)
st.divider()

st.markdown("""
    <div>
        <h1 style="color : #d38856">
            Effect
        </h1>
    </div>""", unsafe_allow_html=True)

st.markdown("""
    <div>
        <h2 style="color : #d38856">Where can this project be applied?</h2>
        <ul>
            <li>If the event point is predicted by the probe, a threshold is set to define a range that includes the seismic waves before and after the event.</li>
            <li>By sending only the seismic wave information from specific intervals that contain event points to Earth, we can reduce the amount of power consumed. Sending only efficient data can minimize the risk of losing important information and enhance the reliability of the data.</li>
        </ul>
        <h2 style="color : #d38856">Benefits of Seismic Wave Exploration</h2>
        <div style="display: flex; align-items: flex-start; justify-content: space-between;">
        <div style="flex: 1; margin-right: 30px;">
            <figure>
                <img src="https://i0.wp.com/eos.org/wp-content/uploads/2022/11/mars-insight-impact-seismic-waves.png?w=1200&ssl=1" style="width: 400px;"/>
                <figcaption style="text-align:center; font-size:14px;color:#555;">Mars Seismic</figcaption>
            </figure>
        </div>
        <div style="flex: 2;">
            
    <p><h3>1. Understanding the Internal Structure of Celestial Bodies</h3></p>
    <p>waves that change as they pass through the interior of celestial bodies, we can understand their internal structure. This involves studying the refraction, reflection, and speed reduction of seismic waves.</p>
        </div>
    </div>
    <p><h3>2. Providing Basic Data for Exploration</h3></p>
    <p>Collecting seismic wave data from Mars can provide foundational data for future Mars exploration missions. This data will enable a sufficient understanding of the crust, which is essential for constructing stable structures or for exploration robots in future missions.</p>
    <p><h3>3. Researching Planetary Evolution and Formation Processes</h3></p>
    <p>By comparing seismic wave data from various celestial bodies, we can understand their evolution and formation processes. This allows us to gain insights into past geological activities of each body and assess the presence of current geological activities, as well as understand rock structures and the flow of internal heat.</p>
    </div>
""", unsafe_allow_html=True)
