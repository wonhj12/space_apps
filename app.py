import streamlit as st

st.set_page_config(page_title="NASA Apps", layout="wide")

# 페이지 선택
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["Home", "Page 1", "Page 2"])

# 선택된 페이지에 따라 내용 표시
if page == "Home":
    import pages.home
elif page == "Page 1":
    import pages.predict
elif page == "Page 2":
    import pages.purpose
