# %% Import Libraries

import streamlit as st
from eda_analysis import clean_feature_name
from st_aggrid import AgGrid
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import RFE

from lightgbm import LGBMRegressor

def clean_processed_feature(feature: str) -> str:
    feature_cleaned = (
        feature.replace("numeric__", "")
        .replace("binary__", "")
        .replace("categorical__", "")
    )
    feature_cleaned = " ".join([f.capitalize() for f in feature_cleaned.split("_")])
    return feature_cleaned

# %% Setup the data

data = st.session_state.data
response_feature = st.session_state.response_feature
response_feature_cleaned = clean_feature_name(response_feature)
data_processing = st.session_state.data_processing
X = data.drop(columns=[response_feature])
y = data[response_feature]
saved_models = st.session_state.saved_models



# %% User Interaction

for model_name, model in saved_models.items():
    st.subheader(f"Model: {model_name}")
    st.write(model)
