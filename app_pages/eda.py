# %% Import Libraries

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

from app_pages.util.utilities import get_default_data_raw

# %% Page Setup

# st.set_page_config(layout='wide')
# col1, col2, col3 = st.columns([2, 5, 2])
st.set_page_config(layout="centered")
st.title("Exploratory Data Analysis")

introduction_text = """
The goal of this page is to explore the speed dating dataset and understand the relationships between various features and the match rate.
"""
st.markdown(introduction_text)

st.subheader("Explanation of the Data")

explanation_text = """
The original dataset consists of 8,000 speed dates, with each date involving two participants.
For most of this analysis, we will focus on the individual level, summarizing each person's responses across all their dates:  
-- match rate: the fraction of dates where the partner was interested in the participant  
-- age: the age of the participant  
-- college SAT score: the median SAT score of the college the participant attended, as a proxy for intelligence  
-- income zip code: the median income of the zip code the participant lived in, as a proxy for wealth  
-- date frequency: how often the participant goes on dates  
-- go out: how often the participant goes out  
-- expect happy: how happy the participant expects to be with the people they meet, taken before the event  
-- expect matches: how many people the participant thinks will be interested in them, taken before the event  
-- estimated matches: how many matches the participant estimates, taken at the end of the night  
-- satisfaction: how satisfied the participant was with the people they met, taken the day after  

There are also six attributes that are expected to be important in dating (Attractive, Ambitious, Fun, Intelligent, Shared Interests, and Sincere),
measured in multiple ways, depending on the perspective. For simplicity, we assign one-word labels to each perspective:  
-- Looking: What participants look for in a partner  
-- Same: What participants think their same sex peers look for  
-- Opposite: What participants think the opposite gender looks for  
-- Yourself: How the participants rates themselves  
-- Partner: How the participant's partner rates them averaged across all dates  
"""
st.markdown(explanation_text)

# %% Load and preprocess data

individual_summary = get_default_data_raw().copy()
# convert gender from bool to string, true is male, false is female
individual_summary["Gender"] = (
    individual_summary["Gender"]
    .astype(str)
    .replace({"True": "Male", "False": "Female"})
)

# list of features
features = individual_summary.columns.to_list()
features.remove("Gender")

# Average attributes by gender
attribute_male = (
    individual_summary[individual_summary["Gender"] == "Male"][features]
    .mean()
    .reset_index()
)
attribute_male.columns = ["attribute", "value"]
attribute_female = (
    individual_summary[individual_summary["Gender"] == "Female"][features]
    .mean()
    .reset_index()
)
attribute_female.columns = ["attribute", "value"]

# %% Bar and Histogram Plots


def bar_and_histogram_plot(
    data: pd.DataFrame, feature: str, category: str, palette: dict = None
) -> plt.Figure:
    """Create a bar plot and histogram for a given feature and category in the data.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        feature (str): The feature to plot on the x-axis.
        category (str): The category to plot on the y-axis.
        palette (dict): A dictionary mapping categories to colors for the bar plot.

    Returns:
        plt.Figure: The figure containing the bar plot and histogram.
    """

    fig_bar, (ax_bar, ax_hist) = plt.subplots(2, 1, sharex=True)
    ax_bar.set_title(f"Bar and Histogram Plots for {feature}")
    sns.barplot(
        data=data,
        x=feature,
        y=category,
        hue=category,
        orient="h",
        ax=ax_bar,
        palette=palette,
        errorbar=None,
    )
    # add bar labels with mean values
    ax_bar.bar_label(ax_bar.containers[0], fmt="%.2f", padding=3)
    ax_bar.bar_label(ax_bar.containers[1], fmt="%.2f", padding=3)
    ax_bar.set_ylabel(category)

    sns.histplot(data, x=feature, hue=category, ax=ax_hist, palette=palette)
    ax_hist.set_xlabel(feature)
    return fig_bar


st.subheader("Compare Features by Gender")
feature = st.selectbox(label="Select Feature to Compare", options=features, index=0)
palette = {"Female": "red", "Male": "blue"}
fig_bar = bar_and_histogram_plot(
    individual_summary, feature=feature, category="Gender", palette=palette
)
st.pyplot(fig_bar)

# %% Attribute Plots


def plot_attribute(
    attribute_male: pd.DataFrame,
    attribute_female: pd.DataFrame,
    title: str,
    identifier: str,
) -> go.Figure:
    """Create a radar plot comparing attributes of male and female participants.

    Parameters:
        attribute_male (pd.DataFrame): DataFrame containing male attributes.
        attribute_female (pd.DataFrame): DataFrame containing female attributes.
        title (str): Title of the plot.
        identifier (str): Identifier to filter attributes (e.g., '_looking', '_same', '_opposite', '_yourself', '_others').

    Returns:
        go.Figure: The radar plot comparing male and female attributes.
    """
    fig = go.Figure()
    data = attribute_female[attribute_female.attribute.str.contains(identifier)]
    # Remove the identifier from the attribute names, replace underscore with space and capitalize
    attributes = (
        data["attribute"]
        .str.replace(identifier, "")
        .str.replace("_", " ")
        .str.capitalize()
        .tolist()
    )
    fig.add_trace(
        go.Scatterpolar(
            r=data["value"],
            theta=attributes,
            fill="toself",
            name="Female",
            marker=dict(color="red"),
        )
    )
    data = attribute_male[attribute_male.attribute.str.contains(identifier)]
    fig.add_trace(
        go.Scatterpolar(
            r=data["value"],
            theta=attributes,
            fill="toself",
            name="Male",
            marker=dict(color="blue"),
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, showticklabels=False)),
        title_text=title,
    )
    return fig


st.subheader("Compare Attributes Across Different Perspectives")
# perspectives = ["Looking", "Same", "Opposite", "Yourself", "Partner"]
perspectives = {
    "Looking": "Participants look for",
    "Same": "Same sex peers look for",
    "Opposite": "Opposite gender looks for",
    "Yourself": "Rates themselves",
    "Partner": "Partner's rating of them",
}

attribute_column1, attribute_column2 = st.columns(2)
with attribute_column1:
    perspective_selected1 = st.selectbox(
        label="Select Perspective", options=perspectives.keys(), index=1
    )
    fig_attribute1 = plot_attribute(
        attribute_male,
        attribute_female,
        title=perspectives[perspective_selected1],
        identifier=f" {perspective_selected1}",
    )
    st.plotly_chart(fig_attribute1, key="attribute1")
with attribute_column2:
    perspective_selected2 = st.selectbox(
        label="Select Perspective", options=perspectives.keys(), index=2
    )
    fig_attribute2 = plot_attribute(
        attribute_male,
        attribute_female,
        title=perspectives[perspective_selected2],
        identifier=f" {perspective_selected2}",
    )
    st.plotly_chart(fig_attribute2, key="attribute2")


# %% Features that influence match making


def regression_plot(
    individual_summary: pd.DataFrame, gender: str, x_feature: str, y_feature: str
) -> plt.Figure:
    """Create a regression plot for two features in the individual summary data, filtered by gender.

    Parameters:
        individual_summary (pd.DataFrame): The DataFrame containing individual summary data.
        gender (str): The gender to filter by ('Male', 'Female', or 'All').
        x_feature (str): The feature to plot on the x-axis.
        y_feature (str): The feature to plot on the y-axis.

    Returns:
        plt.Figure: The figure containing the regression plot with a correlation coefficient.
    """
    if gender == "Female":
        data = individual_summary[individual_summary["Gender"] == "Female"]
    elif gender == "Male":
        data = individual_summary[individual_summary["Gender"] == "Male"]
    else:
        data = individual_summary
    data = data.drop("Gender", axis=1)
    corr_matrix = data.corr()

    # Plot a scatterplot with regression line and correlation coefficient
    plt.figure()
    sns.regplot(x=x_feature, y=y_feature, data=data, scatter_kws={"alpha": 0.5})
    plt.title(f"Correlation: {corr_matrix[x_feature][y_feature]:.2f}")
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.grid(True)
    plt.show()

    return plt


st.subheader("Regression Analysis of Features Influencing Match Rate")
response_feature = "Match Rate"
genders = ["All", "Female", "Male"]
regression_features = features.copy()
regression_features.remove(response_feature)

regression_feature = st.selectbox(
    label="Select Feature to Compare",
    options=regression_features,
    index=regression_features.index("Attractive Partner"),
)
gender_selected = st.selectbox(label="Gender", options=genders, index=0)
fig_regression = regression_plot(
    individual_summary,
    gender=gender_selected,
    x_feature=regression_feature,
    y_feature=response_feature,
)
st.pyplot(fig_regression)
