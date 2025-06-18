# %% Import Libraries

import streamlit as st

from preprocessing import initialize_session_state

# .\virtual-environments\love-env\Scripts\activate
# streamlit run streamlit_app.py

# %% Initialize data processing session state

_data_raw = initialize_session_state()


# %% Page Navigation

if "saved_models" not in st.session_state:
    pages = [
        st.Page(page="home.py", title="Home"),
        st.Page(page="eda.py", title="Exploratory Data Analysis"),
        st.Page(page="preprocessing.py", title="Data Preprocessing"),
        st.Page(page="modeling.py", title="Modeling"),
    ]
else:
    pages = [
        st.Page(page="home.py", title="Home"),
        st.Page(page="eda.py", title="Exploratory Data Analysis"),
        st.Page(page="preprocessing.py", title="Data Preprocessing"),
        st.Page(page="modeling.py", title="Modeling"),
        st.Page(page="model_comparison.py", title="Model Comparison"),
        # st.Page(page="test.py", title="Test Saved Models"),
    ]

pg = st.navigation(pages=pages)
pg.run()
