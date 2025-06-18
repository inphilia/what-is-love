
# %% Import Libraries
import pandas as pd
from helper import load_data_from_file
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# %matplotlib inline

# %% Load the Data from files

@st.cache_data
def get_processed_data(gender_to_string_flag: bool = True) -> pd.DataFrame:
    speed_dating_data = load_data_from_file(file_name='speed_dating_data_cleaned.pkl')
    # join the partner's reponses
    speed_dating_data = pd.merge(
        speed_dating_data,
        speed_dating_data,
        left_on=['pid', 'iid'],
        right_on=['iid', 'pid'],
        suffixes=('', '_partner'),
        how='left'
    )

    # Summarize an individual
    # # group by iid and get the first age, gender and income, and the mean decision_partner
    individual_summary = speed_dating_data.groupby('iid').agg({
        'decision_partner': 'mean',
        'age': 'first',
        'gender': 'first',
        'college_sat_score': 'first',  # SAT score
        'income_zip_code': 'first', # income based on zip code
        'date_frequency': 'first', # how often do you go on dates
        'go_out': 'first',  # how often do you go out
        'expect_happy': 'first',  # how happy do you expect to be with the people, taken before the event
        'expect_matches': 'first',  # how many people do you think you'll be interested in, taken before the event
        'estimated_matches': 'first',  # how many matches do you estimate, taken at the end of the night
        'satisfaction': 'first',  # how satisfied were you with the people you met, taken the day after

        # what you look for in a partner
        'attractive_looking': 'first',
        'ambitious_looking': 'first',
        'fun_looking': 'first',
        'intelligent_looking': 'first',
        'shared_interests_looking': 'first',
        'sincere_looking': 'first',

        # what you think your same gender looks for
        'attractive_same': 'first',
        'ambitious_same': 'first',
        'fun_same': 'first',
        'intelligent_same': 'first',
        'shared_interests_same': 'first',
        'sincere_same': 'first',

        # what you think the opposite gender looks for
        'attractive_opposite': 'first',
        'ambitious_opposite': 'first',
        'fun_opposite': 'first',
        'intelligent_opposite': 'first',
        'shared_interests_opposite': 'first',
        'sincere_opposite': 'first',

        # how do you rate yourself
        'attractive_yourself': 'first',
        'ambitious_yourself': 'first',
        'fun_yourself': 'first',
        'intelligent_yourself': 'first',
        'sincere_yourself': 'first',

        # how do you think others rate you
        'attractive_others': 'first',
        'ambitious_others': 'first',
        'fun_others': 'first',
        'intelligent_others': 'first',
        'sincere_others': 'first',

        # how did your partner rate you
        'attractive_partner': 'mean',
        'ambitious_partner': 'mean',
        'fun_partner': 'mean',
        'intelligent_partner': 'mean',
        'shared_interests_partner': 'mean',
        'sincere_partner': 'mean',
    }).reset_index()
    individual_summary = individual_summary.rename(columns={'decision_partner': 'match_rate'})
    individual_summary['match_rate'] = pd.to_numeric(individual_summary['match_rate'], errors='coerce')
    if gender_to_string_flag:
        # convert gender from bool to string, true is male, false is female
        individual_summary['gender'] = individual_summary['gender'].astype(str).replace({'True': 'Male', 'False': 'Female'})

    return individual_summary


# %% Bar and Histogram Plots

def clean_feature_name(feature: str) -> str:
    """Clean the feature name for display."""
    # return feature.replace('_', ' ').capitalize()
    return ' '.join([f.capitalize() for f in feature.split('_')])


def bar_and_histogram_plot(data: pd.DataFrame, feature: str, category: str, palette: dict = None):

    feature_clean = clean_feature_name(feature)
    category_clean = clean_feature_name(category)

    fig_bar, (ax_bar, ax_hist) = plt.subplots(2, 1, sharex=True)
    ax_bar.set_title(f'Bar and Histogram Plots for {feature_clean}')
    sns.barplot(data=data, x=feature, y=category, hue=category, orient='h', ax=ax_bar, palette=palette, errorbar=None)
    # add bar labels with mean values
    ax_bar.bar_label(ax_bar.containers[0], fmt='%.2f', padding=3)
    ax_bar.bar_label(ax_bar.containers[1], fmt='%.2f', padding=3)
    ax_bar.set_ylabel(category_clean)

    sns.histplot(data, x=feature, hue=category, ax=ax_hist, palette=palette)
    ax_hist.set_xlabel(feature_clean)
    return fig_bar, ax_bar, ax_hist

# %% Attribute plots

def plot_attribute(attribute_male: pd.DataFrame, attribute_female: pd.DataFrame, title: str, identifier: str):
    fig = go.Figure()
    data = attribute_female[attribute_female.attribute.str.contains(identifier)]
    # Remove the identifier from the attribute names, replace underscore with space and capitalize
    attributes = data['attribute'].str.replace(identifier, '').str.replace('_', ' ').str.capitalize().tolist()
    fig.add_trace(go.Scatterpolar(
        r=data['value'],
        theta=attributes,
        fill='toself',
        name='Female',
        marker=dict(color='red')
    ))
    data = attribute_male[attribute_male.attribute.str.contains(identifier)]
    fig.add_trace(go.Scatterpolar(
        r=data['value'],
        theta=attributes,
        fill='toself',
        name='Male',
        marker=dict(color='blue')
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                showticklabels=False
            )
        ),
        title_text=title
    )
    # fig.show()
    return fig

# %% Regression Plots

def regression_plot(individual_summary: pd.DataFrame, gender: str, x_feature: str, y_feature:str):
    
    if gender == 'Female':
        data = individual_summary[individual_summary.gender == 'Female']
    elif gender == 'Male':
        data = individual_summary[individual_summary.gender == 'Male']
    else:
        data = individual_summary
    data = data.drop('gender', axis=1)
    corr_matrix = data.corr()

    x_feature_cleaned = clean_feature_name(x_feature)
    y_feature_cleaned = clean_feature_name(y_feature)

    # Plot a scatterplot with regression line and correlation coefficient
    plt.figure()
    sns.regplot(x=x_feature, y=y_feature, data=data, scatter_kws={'alpha': 0.5})
    # plt.title(f'Scatterplot of {x_feature_cleaned} vs {y_feature_cleaned}\nCorrelation: {corr_matrix[x_feature][y_feature]:.2f}')
    plt.title(f'Correlation: {corr_matrix[x_feature][y_feature]:.2f}')
    plt.xlabel(x_feature_cleaned)
    plt.ylabel(y_feature_cleaned)
    plt.grid(True)
    plt.show()

    return plt

# %% Quantile Bar Plot

