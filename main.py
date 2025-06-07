# %% Import Libraries
import pandas as pd
import numpy as np
import os
from ydata_profiling import ProfileReport
import streamlit as st
from streamlit_pandas_profiling import st_profile_report

# %% Load the Data
data_folder = 'data'
speed_dating_file = 'Speed Dating Data.csv'

speed_dating_data = pd.read_csv(os.path.join(data_folder, speed_dating_file), encoding="ISO-8859-1")
speed_dating_data.head()

# %% 

# Remove columns with more than threshold missing values
missing_threshold = 0.5
missing_percentage = speed_dating_data.isna().sum() / speed_dating_data.shape[0] * 100
columns_to_drop = missing_percentage[missing_percentage > missing_threshold].index
print(f"Dropping columns with more than {missing_threshold * 100}% missing values: {columns_to_drop.tolist()}")



# %% Data Preprocessing

# @st.cache_resource
# def profile_report(data: pd.DataFrame) -> ProfileReport:
#     """
#     Generate a profile report for the given DataFrame.
    
#     Parameters:
#     data (pd.DataFrame): The DataFrame to profile.
    
#     Returns:
#     ProfileReport: The generated profile report.
#     """
#     return ProfileReport(data, explorative=True)

# speed_dating_profile = profile_report(speed_dating_data)
# st_profile_report(speed_dating_profile)

# %%





