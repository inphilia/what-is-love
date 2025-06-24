"""Streamlit Application

This is the main file for setting up the Streamlit application for analyzing speed dating data.
It initializes the main dataset, and sets up the page navigation.
"""

# %% Import Libraries

import streamlit as st
import os

from app_pages.util.utilities import initialize_default_data_processing
from app_pages.modeling import initialize_default_modeling

# streamlit run streamlit_app.py
# .\virtual-environments\love-env\Scripts\activate
# pip freeze > requirements.txt

# %% Initialize data processing session state

# Initialize the session state here so that user can go stright to modeling
initialize_default_data_processing()
initialize_default_modeling()


# %% Page Navigation

page_folder = "app_pages"
pages = [
    st.Page(page=os.path.join(page_folder, "home.py"), title="Home", icon="🏠"),
    st.Page(
        page=os.path.join(page_folder, "eda.py"), title="Exploratory Data Analysis", icon="📊"
    ),
    st.Page(
        page=os.path.join(page_folder, "preprocessing.py"), title="Data Preprocessing", icon="🔧"
    ),
    st.Page(page=os.path.join(page_folder, "modeling.py"), title="Modeling", icon="🤖"),
    st.Page(
        page=os.path.join(page_folder, "model_performance.py"),
        title="Model Performance", icon="📈"
    ),
    # st.Page(page=os.path.join(page_folder, "test.py"), title="Test Page"),
]

# For some reason there's a bug where the first time this page redirects to modeling
# if "saved_models" in st.session_state:
#     pages.append(st.Page(page=os.path.join(page_folder, "model_comparison.py"), title="Model Comparison"))

pg = st.navigation(pages=pages)
pg.run()
