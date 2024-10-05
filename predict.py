import streamlit as st

st.set_page_config(page_title="NASA Apps", layout="wide")

# 페이지 선택
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["purpose","working process", "our team"])

# 선택된 페이지에 따라 내용 표시
if page == "purpose":
    import pages.purpose
elif page == "working process":
    import pages.working_process
elif page == "our team":
    import pages.our_team

