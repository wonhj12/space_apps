import streamlit as st

st.title("Challenge Purpose")

# st.image("https://d2pn8kiwq2w21t.cloudfront.net/images/missionswebPIA22743-16_rfbG1OZ.2e16d0ba.fill-548x400-c50.jpg",width=300)
# st.image("https://astrobiology.nasa.gov/uploads/filer_public_thumbnails/filer_public/81/38/81385e55-d3fa-4b54-9f23-0480074d9dcd/apollo_seismometer.jpg__1240x510_q85_subsampling-2.jpg")
# col1, col2 = st.columns([2,1])
# with col1:
#     st.image("https://d2pn8kiwq2w21t.cloudfront.net/images/missionswebPIA22743-16_rfbG1OZ.2e16d0ba.fill-548x400-c50.jpg",width=300)
# with col2:
#     st.write("")

st.markdown("### **Background**")
# st.image("https://d2pn8kiwq2w21t.cloudfront.net/images/missionswebPIA22743-16_rfbG1OZ.2e16d0ba.fill-548x400-c50.jpg",width=300)

st.markdown("""
    <div style="border: 2px solid #001261; border-radius: 10px; padding: 20px; background-color: rgba(249, 249, 249, 0.4); margin-top: 10px;">
        <h4 style="color: black;">Importance of Collecting Seismic Waves from Celestial Bodies</h4>
        <p style="color: black;">Seismic waves travel at different speeds and shapes as they pass through various materials within a planet. This provides a method to study the internal structure of the planet using seismic waves.</p>
        <p style="color: black;">Particularly in the case of Mars, it provides insights into how all rocky planets, including Earth, were formed.</p>
        <img src="https://d2pn8kiwq2w21t.cloudfront.net/images/E1-PIA24761-Seismogram_from_Mars.width-1320.jpg" alt="Seismogram from Mars"style="width: 50%;"/>   
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="border: 2px solid #001261; border-radius: 10px; padding: 20px; background-color: rgba(249, 249, 249, 0.4); margin-top: 30px;">
        <h4 style="color: black;">Problem</h4>
        <p style="color: black; font-weight:bold; font-size:18px; margin-bottom:5px">High Cost</p>
        <p style="color: black;">1. Transmitting the vast amounts of data collected by spacecraft from celestial bodies back to Earth is very costly, as it requires a significant amount of power to send the data, which is greatly affected by distance.</p>
        <p style="color: black;">2. Only some of this data is scientifically useful.</p>
        <img src="images/example_image.png"/>
    </div>
""", unsafe_allow_html=True)
# st.image("")
st.markdown("### **Goal**")
st.write("")
