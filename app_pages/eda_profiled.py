"""
High level summary of the data intended as alternative to ydata profile.

Currently unsused. Originally to display data profile. However rendering the profile takes a long time
and a good solution to speed it up has not been found yet. Seems easy to recreate the basic functionality
with pandas and streamlit. But that's for another time.
"""

# %% Import Libraries
import streamlit as st
from st_aggrid import AgGrid
# import pandas as pd
from pages.utilities import clean_feature_name, get_processed_data

# %% Set Page Config

st.set_page_config(layout="wide")
st.title("Basic EDA")

# %% Process Data

individual_summary = get_processed_data()
individual_summary.columns = [
    clean_feature_name(col) for col in individual_summary.columns
]

# %%

st.subheader("Summary of Participants")
# AgGrid(individual_summary.describe().reset_index(), height=300)

# st.write(individual_summary.describe().reset_index())
st.dataframe(individual_summary.describe().reset_index())

# %% Custom Summary Statistics

# missing (%), distinct (%), mean, std
# min, 25%, 50%, 75%, max
# histogram

# for col in individual_summary.columns:

col = "Iid"
# if pd.api.types.is_numeric_dtype(individual_summary[col]):
# nunique

# for string value_counts() % instead of hist


# %%

st.subheader("Participants")
AgGrid(individual_summary, height=500)
