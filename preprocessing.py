# %% Import Libraries

import streamlit as st
from eda_analysis import get_processed_data, clean_feature_name
from st_aggrid import AgGrid
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# %% Define Data Processing Functions

data_type_options = ["Numeric", "Binary", "Categorical", "Drop Feature"]
missing_handling_options = ["Impute", "Drop"]

# Initialize processing options in session state if not already set
@st.cache_data
def initialize_data_processing(data_raw: pd.DataFrame, drop_threshold: int = 25):
    data_processing = {}
    for f in data_raw.columns:
        # if feature has more than drop_threshold % missing values, set to 'Drop Feature'
        keep_feature_flag = data_raw[f].isnull().mean() * 100 < drop_threshold

        # if data type is boolean
        if pd.api.types.is_bool_dtype(data_raw[f]):
            data_processing[f] = {
                "data_type": (
                    data_type_options[1] if keep_feature_flag else "Drop Feature"
                ),
                "missing_handling": missing_handling_options[0],
                "outlier_min": 0,
                "outlier_max": 1,
            }
        # if data type is numeric
        elif pd.api.types.is_numeric_dtype(data_raw[f]):
            data_processing[f] = {
                "data_type": (
                    data_type_options[0] if keep_feature_flag else "Drop Feature"
                ),
                "missing_handling": missing_handling_options[0],
                "outlier_min": data_raw[f].min(),
                "outlier_max": data_raw[f].max(),
            }
        # if data type is categorical
        elif pd.api.types.is_categorical_dtype(data_raw[f]):
            data_processing[f] = {
                "data_type": (
                    data_type_options[2] if keep_feature_flag else "Drop Feature"
                ),
                "missing_handling": missing_handling_options[0],
                "outlier_min": data_raw[f].min(),
                "outlier_max": data_raw[f].max(),
            }
    return data_processing


# apply data processing options to data
@st.cache_data
def apply_data_processing(
    data_raw: pd.DataFrame, data_processing: dict
) -> pd.DataFrame:
    """Apply data processing options to the raw data."""
    data = data_raw.copy()
    for feature in data_raw.columns:
        n_rows = data.shape[0]
        if feature not in data_processing:
            data_processing[feature]["dropped_rows"] = 0
            continue
        data_type = data_processing[feature]["data_type"]
        missing_handling = data_processing[feature]["missing_handling"]
        outlier_min = data_processing[feature]["outlier_min"]
        outlier_max = data_processing[feature]["outlier_max"]

        if data_type == "Drop Feature":
            data = data.drop(columns=[feature])
            data_processing[feature]["dropped_rows"] = 0
            continue

        # handle missing values
        if missing_handling == "Drop":
            # drop rows with missing values in this feature
            data = data[data[feature].notnull()]

        # handle outliers
        if data_type == "Numeric":
            data = data[(data[feature] >= outlier_min) & (data[feature] <= outlier_max)]

        data_processing[feature]["dropped_rows"] = n_rows - data.shape[0]

    return data, data_processing

def initialize_session_state():
    data_raw = get_processed_data(gender_to_string_flag=False)
    data_raw = data_raw.drop(columns=["iid"])  # drop iid column
    # remove columns containing same, opposite, yourself, and others
    data_raw = data_raw.loc[
        :, ~data_raw.columns.str.contains("same|opposite|yourself|others", case=False)
    ]
    data_processing = initialize_data_processing(data_raw)
    data, data_processing = apply_data_processing(data_raw, data_processing)
    # initialize data processing options and apply to data
    if "data_processing" not in st.session_state:
        st.session_state.data_processing = data_processing
    if "data" not in st.session_state:
        st.session_state.data = data
    if "response_feature" not in st.session_state:
        st.session_state.response_feature = data_raw.columns[0]  # default to first feature
    
    return data_raw

# %% Initialize data processing session state

data_raw = initialize_session_state()
data = st.session_state.data
data_processing = st.session_state.data_processing
corr_matrix = data.corr()

# list of features
features = data_raw.columns.to_list()
features_cleaned = [clean_feature_name(f) for f in features]
feature_map = {fc: f for fc, f in zip(features_cleaned, features)}

# %% Select Response Feature

# st.set_page_config(layout='wide')
# col1, col2, col3 = st.columns([2, 5, 2])
st.title("Data Preprocessing")
st.subheader("Feature of Interest")
default_index = (
    features_cleaned.index("Match Rate") if "Match Rate" in features_cleaned else 0
)
response_feature_selected = st.selectbox(
    "Select response feature", options=features_cleaned, index=default_index
)
response_feature = feature_map[response_feature_selected]
st.session_state.response_feature = response_feature


def get_feature_statistics(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Get statistics for a given feature."""
    stats = df[feature].describe().reset_index()
    stats.loc[len(stats)] = ["missing (%)", df[feature].isnull().mean() * 100]
    stats.loc[len(stats)] = ["distinct", df[feature].nunique()]
    stats.columns = ["Statistic", "Value"]
    return stats


response_feature_stats = get_feature_statistics(data_raw, response_feature)
col1, col2 = st.columns([1, 2])
with col1:
    st.write("Response Feature Statistics")
    AgGrid(response_feature_stats, height=325)
with col2:
    # histogram
    st.write(f"Histogram of {response_feature_selected}")
    fig_response_hist, ax_response_hist = plt.subplots()
    ax_response_hist = sns.histplot(data_raw[response_feature])
    plt.xlabel(response_feature_selected)
    plt.ylabel("Count")
    st.pyplot(fig_response_hist)


# %% Inspect Each Feature

# input_features = features_cleaned
# input_features.remove(response_feature_selected)
st.subheader("Inspect Each Feature")
feature_selected = st.selectbox(
    "Select feature to inspect", options=features_cleaned, index=0
)
feature = feature_map[feature_selected]
col1, col2 = st.columns([1, 1])
with col1:
    # select data type
    data_type = st.selectbox(
        "Select data type",
        options=data_type_options,
        index=data_type_options.index(
            st.session_state.data_processing[feature]["data_type"]
        ),
    )
    # how to handle missing values
    missing_handling = st.selectbox(
        "Select missing value handling",
        options=missing_handling_options,
        index=missing_handling_options.index(
            st.session_state.data_processing[feature]["missing_handling"]
        ),
    )
with col2:
    if data_type == "Numeric":
        # outlier range
        outlier_min = st.number_input(
            "Outlier minimum",
            value=st.session_state.data_processing[feature]["outlier_min"],
        )
        outlier_max = st.number_input(
            "Outlier maximum",
            value=st.session_state.data_processing[feature]["outlier_max"],
        )
    else:
        st.write("No outlier range for non-numeric data types")
# update session state
st.session_state.data_processing[feature] = {
    "data_type": data_type,
    "missing_handling": missing_handling,
    "outlier_min": outlier_min,
    "outlier_max": outlier_max,
}
data, data_processing = apply_data_processing(
    data_raw, st.session_state.data_processing
)
corr_matrix = data.corr()
st.session_state.data = data

# select between histogram and regression plot
plot_type = st.selectbox(
    "Select plot type", options=["Histogram", "Regression Plot"], index=0
)
feature_stats = get_feature_statistics(data, feature)
col1, col2 = st.columns([1, 2])
with col1:
    st.write("Feature Statistics")
    AgGrid(feature_stats, height=325, key=f"feature_stats_{feature_selected}")
with col2:
    if plot_type == "Histogram":
        # histogram
        st.write(f"Histogram of {feature_selected}")
        fig_feature_hist, ax_feature_hist = plt.subplots()
        ax_feature_hist = sns.histplot(data[feature])
        # Add vertical lines for outlier range
        if data_type == "Numeric":
            ax_feature_hist.axvline(
                outlier_min, color="red", linestyle="--", label="Outlier Min"
            )
            ax_feature_hist.axvline(
                outlier_max, color="green", linestyle="--", label="Outlier Max"
            )
            ax_feature_hist.legend()
        plt.xlabel(feature_selected)
        plt.ylabel("Count")
        st.pyplot(fig_feature_hist)
    else:
        # regression plot
        fig_feature, ax_feature = plt.subplots()
        ax_feature = sns.regplot(
            x=feature, y=response_feature, data=data, scatter_kws={"alpha": 0.5}
        )
        # Add vertical lines for outlier range
        if data_type == "Numeric":
            ax_feature.axvline(
                outlier_min, color="red", linestyle="--", label="Outlier Min"
            )
            ax_feature.axvline(
                outlier_max, color="green", linestyle="--", label="Outlier Max"
            )
            ax_feature.legend()
        plt.title(f"Correlation: {corr_matrix[feature][response_feature]:.2f}")
        plt.xlabel(feature_selected)
        plt.ylabel(response_feature_selected)
        plt.grid(True)
        st.pyplot(fig_feature)

# convert data processing options to DataFrame for display
st.subheader("Data Processing Summary")
st.write(f"Before: {data_raw.shape[0]} rows, {data_raw.shape[1]} columns")
st.write(f"After: {data.shape[0]} rows, {data.shape[1]} columns")
data_processing_df = pd.DataFrame.from_dict(
    st.session_state.data_processing, orient="index"
).reset_index()
AgGrid(data_processing_df, height=300, key="data_processing_grid")
