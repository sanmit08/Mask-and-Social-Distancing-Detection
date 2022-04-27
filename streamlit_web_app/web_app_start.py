import web_app_fm
import web_app_sd1
import streamlit as st

st.set_page_config(page_title='Covid Norms local', page_icon=None, initial_sidebar_state='auto')

st.set_option('deprecation.showPyplotGlobalUse', False)

PAGES = {
    "Face Mask Detection": web_app_fm,
    "Social Distancing": web_app_sd1
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
