# %% Import Libraries

import streamlit as st
from st_aggrid import AgGrid
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, RepeatedKFold
# from sklearn.metrics import make_scorer, r2_score, mean_squared_error

from eda_analysis import clean_feature_name
# from modeling import clean_processed_feature
from modeling import get_features_from_pipeline

# %% Setup the data

if "saved_models" not in st.session_state:
    st.write("Go to the Modeling page to save models.")

data = st.session_state.data if "data" in st.session_state else None
response_feature = st.session_state.response_feature
response_feature_cleaned = clean_feature_name(response_feature)
data_processing = st.session_state.data_processing
saved_models = st.session_state.saved_models

X = data.drop(columns=[response_feature])
y = data[response_feature]

# %% User Interaction

st.title(f"Compare Saved Models for Predicting {response_feature_cleaned}")
performance_metric_options = ["R2", "MSE"]
performance_metric_map = {
    "R2": {"scoring": "r2", "multiplier": 1},
    "MSE": {"scoring": "neg_mean_squared_error", "multiplier": -1},
}
performance_metric_selected = st.selectbox(
    "Select Performance Metric", options=performance_metric_options, index=0
)

# %% Bar Chart of Model Performance

def performance_metric_comparison(X, y, saved_models, performance_metric_selected):
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    pm = performance_metric_map[performance_metric_selected]["scoring"]
    multiplier = performance_metric_map[performance_metric_selected]["multiplier"]
    model_performances = []
    for model_name, model in saved_models.items():
        scores = cross_val_score(model, X, y, scoring=pm, cv=cv, n_jobs=-1)
        model_performances.append({"Model": model_name, performance_metric_selected: np.mean(scores) * multiplier})

    # Create a DataFrame for plotting
    performance_df = pd.DataFrame(model_performances)

    # Plotting
    fig_pm, ax_pm = plt.subplots()
    ax_pm = sns.barplot(data=performance_df, x="Model", y=performance_metric_selected)
    plt.xlabel("Model")
    plt.ylabel(performance_metric_selected)
    st.pyplot(fig_pm)

st.subheader("Model Performance Comparison")
performance_metric_comparison(X, y, saved_models, performance_metric_selected)

# %% Feature Importance Comparison

def compare_feature_importance(saved_models):
    importances = []
    for model_name, pipeline in saved_models.items():
        pipeline.fit(X, y)  # ensure the model is fitted
        features = get_features_from_pipeline(pipeline, X)
        features.columns = ['Feature', 'Importance']
        features['Model'] = model_name
        features['Importance'] = features['Importance'].values / np.sum(features['Importance'].values) # scale importances to sum to 1
        importances.append(features)

    if not importances:
        st.write("No models with feature importance available.")
        return
    
    # Combine importances from all models
    importances = pd.concat(importances, axis=0).reset_index(drop=True)

    # Plotting
    fig_fi, ax_fi = plt.subplots()
    ax_fi = sns.barplot(data=importances, x="Importance", y="Feature", hue="Model")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    st.pyplot(fig_fi)

st.subheader("Feature Importance Comparison")
compare_feature_importance(saved_models)

# %% Show Prediction

st.subheader("Make Predictions with Saved Models")
model_selected = st.selectbox("Select Model for Prediction", options=list(saved_models.keys()))

def unclean_feature_name(feature: str) -> str:
    feature_uncleaned = '_'.join([f.lower() for f in feature.split(" ")])
    return feature_uncleaned

# create input form
with st.form("prediction_form"):
    prediction_model = saved_models[model_selected]
    prediction_features = get_features_from_pipeline(prediction_model, X, clean_flag=False)['Feature'].tolist()
    # create input fields for each feature
    input_data = {}
    for feature in X.columns:
        if feature in prediction_features:
            feature_cleaned = clean_feature_name(feature)
            if data_processing[feature]["data_type"] == "categorical":
                input_data[feature] = st.slider(f"Input {feature_cleaned}", value=float(X[feature].mean()), min_value=float(X[feature].min()), max_value=float(X[feature].max()), step=1)
            else:
                input_data[feature] = st.slider(f"Input {feature_cleaned}", value=float(X[feature].mean()), min_value=float(X[feature].min()), max_value=float(X[feature].max()))
        else:
            input_data[feature] = 0  # default value if feature not used in model
    input_df = pd.DataFrame([input_data])

    submitted = st.form_submit_button("Predict")
    if submitted:
        prediction = prediction_model.predict(input_df)[0]
        st.write(f"Predicted {response_feature_cleaned}: {prediction:.2f}")
        # show as vertical line on histogram
        fig_pred, ax_pred = plt.subplots()
        ax_pred = sns.histplot(data[response_feature])
        plt.axvline(prediction, color='red', linestyle='--', label='Prediction')
        plt.xlabel(response_feature_cleaned)
        plt.ylabel("Count")
        plt.legend()
        st.pyplot(fig_pred)
    
