"""
Page for comparing saved models by performance metrics and feature importance.
"""

# %% Import Libraries

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, RepeatedKFold

from app_pages.util.utilities import config, initialize_data
from app_pages.modeling import get_features_from_pipeline

performance_metrics = config.modeling.performance_metrics

# %% Page Setup


def ui_model_performance_page(response_feature: str):
    st.title(f"Model Performance for Predicting {response_feature}")
    introduction_text = """
    This page allows you to compare the performance of different saved models on the speed dating dataset. 
    Be sure to have saved models in the Modeling page before proceeding here, or hit the reset button on the 
    Data Preprocessing page to reset the session state and start over.
    """
    st.markdown(introduction_text)
    return


# %% Bar Chart of Model Performance


def performance_metric_comparison(X: pd.DataFrame, y: pd.Series, saved_models: dict):
    """Compare the performance of saved models using cross-validation.

    Parameters:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        saved_models (dict): Dictionary of saved models.
        performance_metric (dict): Dictionary containing the performance metric details.
    """
    st.subheader("Model Performance Comparison")
    performance_metric_text = """
    This section compares the performance of different saved models using cross-validation. 
    You can select the performance metric to use for comparison from the dropdown menu. 
    I'm told bigger is better. Again, this is not real dating advice.
    """
    st.markdown(performance_metric_text)
    performance_metric_selected = st.selectbox(
        "Select Performance Metric", options=performance_metrics.keys(), index=0
    )
    performance_metric = performance_metrics[performance_metric_selected]
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    pm = performance_metric["scoring"]
    multiplier = performance_metric["multiplier"]
    model_performances = []
    for model_name, model in saved_models.items():
        scores = cross_val_score(model, X, y, scoring=pm, cv=cv, n_jobs=-1)
        model_performances.append(
            {
                "Model": model_name,
                performance_metric["name"]: np.mean(scores) * multiplier,
            }
        )
    performance_df = pd.DataFrame(model_performances)  # Create a DataFrame for plotting

    # Plotting
    fig_pm, ax_pm = plt.subplots()
    ax_pm = sns.barplot(data=performance_df, x="Model", y=performance_metric["name"])
    plt.xlabel("Model")
    plt.ylabel(performance_metric["name"])
    st.pyplot(fig_pm)
    return


# %% Feature Importance Comparison


def compare_feature_importance(saved_models: dict, X: pd.DataFrame, y: pd.Series):
    """Compare feature importance across saved models.

    Parameters:
        saved_models (dict): Dictionary of saved models.
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
    """
    st.subheader("Feature Importance Comparison")
    feature_importance_text = """
    This section compares the feature importance across different saved models. Shared 
    features across models will be shown as grouped bars. The imporance values are scaled to sum to 1 for 
    each model.
    """
    st.markdown(feature_importance_text)
    importances = []
    for model_name, pipeline in saved_models.items():
        pipeline.fit(X, y)  # ensure the model is fitted
        features = get_features_from_pipeline(pipeline, X)
        features.columns = ["Feature", "Importance"]
        features["Model"] = model_name
        features["Importance"] = features["Importance"].values / np.sum(
            features["Importance"].values
        )  # scale importances to sum to 1
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


# %% Show Prediction


def ui_show_prediction(
    saved_models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    response_feature: str,
    data_processing: dict,
):
    """Display the prediction UI for the selected model.

    Parameters:
        model_selected (str): The name of the selected model.
        saved_models (dict): Dictionary of saved models.
        X (pd.DataFrame): Features.
        response_feature (str): The response feature to predict.
    """
    st.subheader("Make Predictions with Saved Models")
    prediction_text = """
    This section allows you to make predictions using the saved models.
    """
    st.markdown(prediction_text)
    model_selected = st.selectbox(
        "Select Model for Prediction", options=list(saved_models.keys())
    )
    # create input form
    with st.form("prediction_form"):
        prediction_model = saved_models[model_selected]
        prediction_features = get_features_from_pipeline(prediction_model, X)[
            "Feature"
        ].tolist()
        # create input fields for each feature
        input_data = {}
        for feature in X.columns:
            if feature in prediction_features:
                if data_processing[feature]["data_type"] == "categorical":
                    input_data[feature] = st.slider(
                        f"Input {feature}",
                        value=float(X[feature].mean()),
                        min_value=float(X[feature].min()),
                        max_value=float(X[feature].max()),
                        step=1,
                    )
                else:
                    input_data[feature] = st.slider(
                        f"Input {feature}",
                        value=float(X[feature].mean()),
                        min_value=float(X[feature].min()),
                        max_value=float(X[feature].max()),
                    )
            else:
                input_data[feature] = 0  # default value if feature not used in model
        input_df = pd.DataFrame([input_data])

        submitted = st.form_submit_button("Predict")
        if submitted:
            prediction = prediction_model.predict(input_df)[0]
            st.write(f"Predicted {response_feature}: {prediction:.2f}")
            # show as vertical line on histogram
            fig_pred, ax_pred = plt.subplots()
            ax_pred = sns.histplot(y)
            plt.axvline(prediction, color="red", linestyle="--", label="Prediction")
            plt.xlabel(response_feature)
            plt.ylabel("Count")
            plt.legend()
            st.pyplot(fig_pred)
    return


# %%


def main():
    if len(st.session_state.saved_models) == 0:
        st.write(
            "Error: No saved models found. Please go to the Modeling page to save models."
        )
        return
    saved_models = st.session_state.saved_models
    X, y, data_processing, response_feature = initialize_data()

    ui_model_performance_page(response_feature)
    performance_metric_comparison(X, y, saved_models)
    compare_feature_importance(saved_models, X, y)
    ui_show_prediction(saved_models, X, y, response_feature, data_processing)
    return


if __name__ == "__main__":
    main()
