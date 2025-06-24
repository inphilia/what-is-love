"""
Preprocessing Page for Streamlit App

This page allows users to select features for data preprocessing, including handling missing values,
outliers, and data types. It also provides visualizations for the selected features and response feature.
"""

# %% Import Libraries

import streamlit as st
from st_aggrid import AgGrid
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from app_pages.util.utilities import config, apply_data_processing, initialize_default_data_processing, get_default_data_raw
from app_pages.modeling import reset_modeling_session_state, initialize_default_modeling

data_type_options = config.data_processing.data_type_options
missing_handling_options = config.data_processing.missing_handling_options
model_type_options = config.modeling.model_type_options

# %% Session State Functions

def update_data_processing_session_state(feature: str, data_processing_feature: dict, data_raw: pd.DataFrame):
    """Update the session state for data processing and response feature.

    Parameters:
        data_processing (dict): The data processing options to update.
        response_feature (str): The response feature to update.
    """
    if data_processing_feature != st.session_state.data_processing[feature]:
        st.session_state.data_processing[feature] = data_processing_feature
        data, dropped_rows = apply_data_processing(data_raw, st.session_state.data_processing)
        st.session_state.data = data
        reset_modeling_session_state()
        return data, st.session_state.data_processing
    return st.session_state.data, st.session_state.data_processing

def update_response_feature_session_state(response_feature: str):
    if response_feature != st.session_state.response_feature:
        st.session_state.response_feature = response_feature
        reset_modeling_session_state()
    return

# %% Setup the page

def ui_introduction():
    st.title("Data Preprocessing")

    introduction_text = """
    This page allows you to select options for data preprocessing, including handling missing values, outliers, 
    and data types. It also provides visualizations to aid decision-making. It pre-populates the options based
    on some basic analysis of the data, but you can change them as needed.
    """
    st.markdown(introduction_text)

    st.subheader("Reset Data Processing Options")
    st.write("This will reset all data processing options to their default values.")
    reset_button = st.button("Reset Data Processing Options")
    if reset_button:
        initialize_default_data_processing()
        initialize_default_modeling()
        st.write("Data processing options have been reset to default values.")
    return

# %% Select Response Feature

def ui_select_response_feature(features: list):
    st.subheader("Feature of Interest")
    response_feature_text = """
    Select the response feature that you want to predict. This is the target variable for your modeling efforts. 
    The default response feature is 'Match Rate', which is the percentage of matches made during the speed dating 
    event."""
    st.markdown(response_feature_text)

    default_index = (
        features.index("Match Rate") if "Match Rate" in features else 0
    )
    response_feature = st.selectbox(
        "Select response feature", options=features, index=default_index
    )
    update_response_feature_session_state(response_feature)
    return response_feature

def get_feature_statistics(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Get statistics for a given feature."""
    stats = df[feature].describe().reset_index()
    stats.loc[len(stats)] = ["missing (%)", df[feature].isnull().mean() * 100]
    stats.loc[len(stats)] = ["distinct", df[feature].nunique()]
    stats.columns = ["Statistic", "Value"]
    return stats

def ui_response_feature_statistics(data_raw: pd.DataFrame, response_feature: str):
    response_feature_stats = get_feature_statistics(data_raw, response_feature)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("Response Feature Statistics")
        AgGrid(response_feature_stats, height=325)
    with col2:
        # histogram
        st.write(f"Histogram of {response_feature}")
        fig_response_hist, ax_response_hist = plt.subplots()
        ax_response_hist = sns.histplot(data_raw[response_feature])
        plt.xlabel(response_feature)
        plt.ylabel("Count")
        st.pyplot(fig_response_hist)
    return


# %% Inspect Each Feature

def show_all_features(data_raw: pd.DataFrame):
    features = data_raw.columns.to_list()
    data_profile_summary = pd.DataFrame(
        {
            "Feature": features,
            "Data Type": [str(data_raw[f].dtype) for f in features],
            "Missing (%)": [f"{data_raw[f].isnull().sum()} ({data_raw[f].isnull().mean() * 100:.1f}%)" for f in features],
            "Distinct (%)": [f"{data_raw[f].nunique()} ({data_raw[f].nunique() / data_raw.shape[0] * 100:.1f}%)" for f in features],
            "Mean": [f"{data_raw[f].mean():.3f}" for f in features],
            "Std": [f"{data_raw[f].std():.3f}" for f in features],
            "Min": [data_raw[f].astype(float).min() for f in features],
            "25%": [data_raw[f].astype(float).quantile(0.25) for f in features],
            "50%": [data_raw[f].astype(float).quantile(0.5) for f in features],
            "75%": [data_raw[f].astype(float).quantile(0.75) for f in features],
            "Max": [data_raw[f].astype(float).max() for f in features]
        }
    )
    # https://pandas.pydata.org/docs/user_guide/style.html#Styler-Functions
    AgGrid(data_profile_summary, height=500, key="data_profile_summary")
    return

def show_data_processing(feature: str, data_processing_feature: dict) -> dict:
    with st.form(key=f"data_processing_form_{feature}"):
        col1, col2 = st.columns([1, 1])
        with col1:
            # select data type
            data_type = st.selectbox(
                "Select data type",
                options=data_type_options,
                index=data_type_options.index(
                    data_processing_feature["data_type"]
                ),
            )
            # how to handle missing values
            missing_handling = st.selectbox(
                "Select missing value handling",
                options=missing_handling_options,
                index=missing_handling_options.index(
                    data_processing_feature["missing_handling"]
                ),
            )
        with col2:
            if data_type == "Numeric":
                # outlier range
                outlier_min = st.number_input(
                    "Outlier minimum",
                    value=data_processing_feature["outlier_min"],
                )
                outlier_max = st.number_input(
                    "Outlier maximum",
                    value=data_processing_feature["outlier_max"],
                )
            else:
                outlier_min = data_processing_feature["outlier_min"]
                outlier_max = data_processing_feature["outlier_max"]
                st.write("No outlier range for non-numeric data types")
        # submit button
        submitted = st.form_submit_button("Apply Data Processing")
        if submitted:
            return {
                "data_type": data_type,
                "missing_handling": missing_handling,
                "outlier_min": outlier_min,
                "outlier_max": outlier_max,
            }

    return data_processing_feature

def plot_feature_histogram(data_raw: pd.DataFrame, feature: str, data_processing_feature: dict):
    # histogram
    st.write(f"Histogram of {feature}")
    fig_feature_hist, ax_feature_hist = plt.subplots()
    ax_feature_hist = sns.histplot(data_raw[feature])
    # Add vertical lines for outlier range
    if data_processing_feature["data_type"] == "Numeric":
        ax_feature_hist.axvline(
            data_processing_feature["outlier_min"], color="red", linestyle="--", label="Outlier Min"
        )
        ax_feature_hist.axvline(
            data_processing_feature["outlier_max"], color="green", linestyle="--", label="Outlier Max"
        )
        ax_feature_hist.legend()
    plt.xlabel(feature)
    plt.ylabel("Count")
    st.pyplot(fig_feature_hist)
    return

def plot_feature_regression(data_raw: pd.DataFrame, feature: str, data_processing_feature: dict, response_feature: str):
    # regression plot
    corr_matrix = data_raw.corr()
    fig_feature, ax_feature = plt.subplots()
    ax_feature = sns.regplot(
        x=feature, y=response_feature, data=data_raw, scatter_kws={"alpha": 0.5}
    )
    # Add vertical lines for outlier range
    if data_processing_feature["data_type"] == "Numeric":
        ax_feature.axvline(
            data_processing_feature["outlier_min"], color="red", linestyle="--", label="Outlier Min"
        )
        ax_feature.axvline(
            data_processing_feature["outlier_max"], color="green", linestyle="--", label="Outlier Max"
        )
        ax_feature.legend()
    plt.title(f"Correlation: {corr_matrix[feature][response_feature]:.2f}")
    plt.xlabel(feature)
    plt.ylabel(response_feature)
    plt.grid(True)
    st.pyplot(fig_feature)
    return

def show_feature(data_raw: pd.DataFrame, feature: str, data_processing_feature: dict, response_feature: str):
    # select between histogram and regression plot
    plot_type = st.selectbox(
        "Select plot type", options=["Histogram", "Regression Plot"], index=0
    )
    feature_stats = get_feature_statistics(data_raw, feature)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("Feature Statistics")
        AgGrid(feature_stats, height=325, key=f"feature_stats_{feature}")
    with col2:
        if plot_type == "Histogram":
            plot_feature_histogram(data_raw, feature, data_processing_feature)
        else:
            plot_feature_regression(data_raw, feature, data_processing_feature, response_feature)
    return

def ui_data_processing(data_raw: pd.DataFrame, data_processing: dict, data: pd.DataFrame, response_feature: str):
    features = data_raw.columns.to_list()
    st.subheader("Inspect Each Feature")
    inspect_text = """
    Select a feature to inspect its data type, missing value handling, and outlier range."""
    st.markdown(inspect_text)
    feature = st.selectbox(
        "Select feature to inspect", options=['All'] + features, index=0
    )

    if feature == 'All':
        show_all_features(data_raw)
    else:
        data_processing_feature = show_data_processing(feature, data_processing[feature])
        data, data_processing = update_data_processing_session_state(feature, data_processing_feature, data_raw)
        if feature in data.columns and data.shape[0] > 0:
            show_feature(data_raw, feature, data_processing[feature], response_feature)
    return data, data_processing


# %% Data Processing Summary


def ui_data_processing_summary(data_raw: pd.DataFrame, data: pd.DataFrame):
    st.subheader("Data Processing Summary")
    if data.shape[0] == 0: # check if data has rows
        st.error("No data available after applying data processing. Please adjust your settings.")
        st.stop()
    else:
        # convert data processing options to DataFrame for display
        st.write(f"Before: {data_raw.shape[0]} rows, {data_raw.shape[1]} columns")
        st.write(f"After: {data.shape[0]} rows, {data.shape[1]} columns")
        data_processing_df = pd.DataFrame.from_dict(
            st.session_state.data_processing, orient="index"
        ).reset_index()
        AgGrid(data_processing_df, height=300, key="data_processing_grid")

# %%

def main():
    # Initialize data processing session state
    data_raw = get_default_data_raw()
    data = st.session_state.data
    data_processing = st.session_state.data_processing

    ui_introduction()
    response_feature = ui_select_response_feature(data_raw.columns.to_list())
    ui_response_feature_statistics(data_raw, response_feature)

    # data processing
    data, data_processing = ui_data_processing(data_raw, data_processing, data, response_feature)
    ui_data_processing_summary(data_raw, data)

if __name__ == "__main__":
    main()

