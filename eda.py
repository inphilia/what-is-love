# %% Import Libraries

import streamlit as st
from eda_analysis import (
    get_processed_data,
    clean_feature_name,
    bar_and_histogram_plot,
    plot_attribute,
    regression_plot,
)

# %%

# st.set_page_config(layout='wide')
# col1, col2, col3 = st.columns([2, 5, 2])
st.set_page_config(layout="centered")
st.title("Exploratory Data Analysis")
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

individual_summary = get_processed_data()
# list of features
features = individual_summary.columns.to_list()
features.remove("gender")
features.remove("iid")
features_cleaned = [clean_feature_name(f) for f in features]
feature_map = {fc: f for fc, f in zip(features_cleaned, features)}

# Average attributes by gender
attribute_male = (
    individual_summary[individual_summary.gender == "Male"][features]
    .mean()
    .reset_index()
)
attribute_male.columns = ["attribute", "value"]
attribute_female = (
    individual_summary[individual_summary.gender == "Female"][features]
    .mean()
    .reset_index()
)
attribute_female.columns = ["attribute", "value"]

# %% Bar and Histogram Plots

st.subheader("Compare Features by Gender")
feature_selected = st.selectbox(
    label="Select Feature to Compare", options=features_cleaned, index=0
)
feature = feature_map[feature_selected]
palette = {"Female": "red", "Male": "blue"}
fig_bar, ax_bar, ax_hist = bar_and_histogram_plot(
    individual_summary, feature=feature, category="gender", palette=palette
)
st.pyplot(fig_bar)

# %% Attribute Plots

st.subheader("Compare Attributes Across Different Perspectives")
perspectives = ["looking", "same", "opposite", "yourself", "partner"]
perspectives_cleaned = [p.capitalize() for p in perspectives]
perspective_title = {
    "Looking": "Participants look for",
    "Same": "Same sex peers look for",
    "Opposite": "Opposite gender looks for",
    "Yourself": "Rates themselves",
    "Partner": "Partner's rating of them",
}

attribute_column1, attribute_column2 = st.columns(2)
with attribute_column1:
    perspective_selected1 = st.selectbox(
        label="Select Perspective", options=perspectives_cleaned, index=1
    )
    fig_attribute1 = plot_attribute(
        attribute_male,
        attribute_female,
        title=perspective_title[perspective_selected1],
        identifier=f"_{perspective_selected1.lower()}",
    )
    st.plotly_chart(fig_attribute1, key="attribute1")
with attribute_column2:
    perspective_selected2 = st.selectbox(
        label="Select Perspective", options=perspectives_cleaned, index=2
    )
    fig_attribute2 = plot_attribute(
        attribute_male,
        attribute_female,
        title=perspective_title[perspective_selected2],
        identifier=f"_{perspective_selected2.lower()}",
    )
    st.plotly_chart(fig_attribute2, key="attribute2")


# %% Features that influence match making

st.subheader("Regression Analysis of Features Influencing Match Making")
genders = ["All", "Female", "Male"]
regression_features = features_cleaned.copy()
regression_features.remove("Match Rate")

regression_feature_selected = st.selectbox(
    label="Select Feature to Compare",
    options=regression_features,
    index=regression_features.index("Attractive Partner"),
)
gender_selected = st.selectbox(label="Gender", options=genders, index=0)
regression_feature = feature_map[regression_feature_selected]
fig_regression = regression_plot(
    individual_summary,
    gender=gender_selected,
    x_feature=regression_feature,
    y_feature="match_rate",
)
st.pyplot(fig_regression)
