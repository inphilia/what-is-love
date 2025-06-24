# %% Import Libraries

import streamlit as st
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

from app_pages.util.utilities import config, clean_processed_feature, initialize_data

model_type_options = config.modeling.model_type_options

# %% Combined Pipeline Functions


def clean_feature(feature: str) -> str:
    return " ".join([f.capitalize() for f in feature.split(" ")])


def get_features_from_pipeline(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Extract features and their importances from the fitted pipeline.

    Parameters:
        pipeline (Pipeline): The fitted pipeline containing feature selection and regression steps.
        X (pd.DataFrame): The input features used for fitting the pipeline.

    Returns:
        pd.DataFrame: A DataFrame containing the selected features and their importances or coefficients.
    """
    features_df = pd.DataFrame()
    # lightgbm grid search
    if "regression" not in pipeline.named_steps and "lgbm_grid" in pipeline.named_steps:
        selected_features = X.columns[
            pipeline.named_steps["feature_selection"].get_support()
        ]
        feature_importances = pipeline.named_steps[
            "lgbm_grid"
        ].best_estimator_.feature_importances_
        sorted_index = np.argsort(feature_importances)[::-1]
        feature_importances = feature_importances[sorted_index]
        selected_features = np.array(selected_features)[sorted_index]
        features_df = pd.DataFrame(
            {"Feature": selected_features, "Feature Importance": feature_importances}
        )
    # lightgbm without grid search
    elif "feature_selection" in pipeline.named_steps and hasattr(
        pipeline.named_steps["regression"], "feature_importances_"
    ):
        selected_features = X.columns[
            pipeline.named_steps["feature_selection"].get_support()
        ]
        feature_importances = pipeline.named_steps["regression"].feature_importances_
        sorted_index = np.argsort(feature_importances)[::-1]
        feature_importances = feature_importances[sorted_index]
        selected_features = np.array(selected_features)[sorted_index]
        features_df = pd.DataFrame(
            {"Feature": selected_features, "Feature Importance": feature_importances}
        )
    # linear regression
    elif "preprocessor" in pipeline.named_steps and hasattr(
        pipeline.named_steps["regression"], "coef_"
    ):
        coefficients = pipeline.named_steps["regression"].coef_
        processed_features = pipeline.named_steps[
            "preprocessor"
        ].get_feature_names_out()
        selected_features = [
            clean_processed_feature(f)
            for f in processed_features[
                pipeline.named_steps["feature_selection"].get_support()
            ]
        ]
        selected_features = [clean_feature(feature) for feature in selected_features]
        sorted_index = np.argsort(np.abs(coefficients))[
            ::-1
        ]  # sort by decreasing feature importance based on coef
        coefficients = coefficients[sorted_index]
        selected_features = np.array(selected_features)[sorted_index]
        features_df = pd.DataFrame(
            {"Feature": selected_features, "Coefficient": coefficients}
        )
    return features_df


def train_regression_over_range(
    X: pd.DataFrame, y: pd.Series, data_processing: dict, model_type: dict
) -> dict:
    """Scores pipelines over different number of features based on model type.

    Parameters:
        X (pd.DataFrame): The input features.
        y (pd.Series): The target variable.
        data_processing (dict): Dictionary containing data processing information for each feature.
        model_type (dict): Dictionary containing model type information, including name and scoring method.

    Returns:
        model_given_n_features (dict): Dictionary containing pipelines, scores and selected features
            for each number of features.
    """
    # identify feature types
    numeric_features = []
    binary_features = []
    categorical_features = []
    for col in X.columns:
        if data_processing[col]["data_type"] == "Numeric":
            numeric_features.append(col)
        elif data_processing[col]["data_type"] == "Binary":
            binary_features.append(col)
        elif data_processing[col]["data_type"] == "Categorical":
            categorical_features.append(col)

    steps = []
    if model_type["name"] == "Linear Regression":
        numeric_preprocessor = Pipeline(
            steps=[
                ("imputation_median", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        binary_preprocessor = Pipeline(
            steps=[("imputation_mode", SimpleImputer(strategy="most_frequent"))]
        )
        categorical_preprocessor = Pipeline(
            steps=[
                ("imputation_mode", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_preprocessor, numeric_features),
                ("binary", binary_preprocessor, binary_features),
                ("categorical", categorical_preprocessor, categorical_features),
            ]
        )
        steps.append(("preprocessor", preprocessor))

    estimator = (
        LinearRegression()
        if model_type["name"] == "Linear Regression"
        else LGBMRegressor(verbose=-1)
    )
    scoring = model_type["scoring"]
    rkf = RepeatedKFold(n_splits=2, n_repeats=3, random_state=42)
    model_given_n_features = {}
    for n_features in np.arange(1, len(X.columns) + 1):
        rfe = RFE(estimator, n_features_to_select=n_features)
        pipeline_n = Pipeline(
            steps=steps + [("feature_selection", rfe), ("regression", estimator)]
        )
        scores = cross_val_score(
            pipeline_n,
            X,
            y,
            scoring=scoring,
            cv=rkf,
            n_jobs=-1,
        )
        pipeline_n.fit(X, y)
        model_given_n_features[n_features] = {
            "pipeline": pipeline_n,
            "mean_score": np.mean(scores),
            "features_df": get_features_from_pipeline(pipeline_n, X),
        }

    return model_given_n_features


def optimize_selected_lgbm(
    X: pd.DataFrame, y: pd.Series, n_features_selected: int
) -> Pipeline:
    """Optimize LightGBM model with selected number of features using RFE and GridSearchCV.

    Parameters:
        X (pd.DataFrame): The input features.
        y (pd.Series): The target variable.
        n_features_selected (int): The number of features to select for optimization.

    Returns:
        Pipeline: The optimized pipeline with feature selection and LightGBM regression.
    """
    # build new pipeline with selected features and grid search
    rfe = RFE(LGBMRegressor(verbose=-1), n_features_to_select=n_features_selected)
    lgbm_grid = {
        "num_leaves": [5, 20, 31],
        "learning_rate": [0.05, 0.1, 0.2],
        "n_estimators": [50, 100, 150],
        "max_depth": [2, 4, 6, 8],  # range(2, 10, 2)
    }
    grid_search = GridSearchCV(
        LGBMRegressor(verbose=-1),
        lgbm_grid,
        cv=3,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
    )
    search_pipeline = Pipeline(
        steps=[("feature_selection", rfe), ("lgbm_grid", grid_search)]
    )
    search_pipeline.fit(X, y)

    # use the best parameters to build the final pipeline
    best_params = search_pipeline.named_steps["lgbm_grid"].best_params_
    best_params["verbose"] = -1
    lgbm_optimized = LGBMRegressor(**best_params)
    pipeline_optimized = Pipeline(
        steps=[
            ("feature_selection", rfe),
            ("regression", lgbm_optimized),
        ]
    )
    pipeline_optimized.fit(X, y)

    return pipeline_optimized


# %% User Interaction for Model Selection


def get_models_over_range(
    model_type: dict, X: pd.DataFrame, y: pd.Series, data_processing: dict
) -> dict:
    """Train regression models over a range of feature selections based on the selected model type.

    Parameters:
        model_type_selected (str): The selected model type from the dropdown menu.
        X (pd.DataFrame): The input features.
        y (pd.Series): The target variable.
        data_processing (dict): Dictionary containing data processing information for each feature.

    Returns:
        model_given_n_features (dict): Dictionary containing pipelines, scores, and selected features
            for each number of features.
    """

    if model_type["name"] not in st.session_state.train_models_over_range:
        model_given_n_features = train_regression_over_range(
            X, y, data_processing, model_type
        )
        st.session_state.train_models_over_range[model_type["name"]] = (
            model_given_n_features
        )
    else:
        model_given_n_features = st.session_state.train_models_over_range[
            model_type["name"]
        ]

    return model_given_n_features


def ui_model_scores(model_given_n_features: dict, model_type: dict) -> None:
    """User interface for displaying model scores and selected features based on the number of features.

    Parameters:
        model_given_n_features (dict): Dictionary containing pipelines, scores, and selected features
            for each number of features.
        model_type (dict): Dictionary containing model type information, including name and scoring method.
    """
    st.subheader(f"{model_type['name']} Pipeline")

    # Plot the scores for different numbers of features
    n_features_range = model_given_n_features.keys()
    scores = [model_given_n_features[n]["mean_score"] for n in n_features_range]
    score_name = model_type["scoring_name"]
    pipeline_text = """
    For each number of features selected, a model pipeline is created using RFE (Recursive Feature 
    Elimination) with the specified model type. We wish to select the number of features that maximizes the 
    mean score of the model. The mean score is calculated using a Repeated K-Fold cross-validation.
    """
    st.markdown(pipeline_text)
    fig_lr, ax_lr = plt.subplots()
    ax_lr = sns.lineplot(x=n_features_range, y=scores, marker="o")
    plt.xlabel("Number of Features Selected")
    plt.ylabel(f"Mean {score_name}")
    plt.grid()
    st.pyplot(fig_lr)

    # Display the scores and selected features in a table
    selected_features_text = """
    The following table summarizes the mean scores and selected features for each number of features. It 
    lists the features in order of importance based on the model's feature selection process.
    """
    st.markdown(selected_features_text)
    selected_features_list = [
        ", ".join(model_given_n_features[n]["features_df"]["Feature"])
        for n in n_features_range
    ]
    models_summary = pd.DataFrame(
        {
            "Number of Features": n_features_range,
            f"Mean {score_name}": scores,
            "Selected Features": selected_features_list,
        }
    )
    AgGrid(models_summary, height=300)

    return


# %% Save Optimized Model


def ui_feature_selection(model_given_n_features: dict) -> int:
    """User interface for optimizing the selected model based on the number of features.

    Parameters:
        model_given_n_features (dict): Dictionary containing pipelines, scores, and selected features
    """
    # Optimization based on user-selected number of features
    st.subheader("Model Optimization")
    # Get the number of features with the maximum mean score
    default_n_features = max(
        model_given_n_features.keys(),
        key=lambda k: model_given_n_features[k]["mean_score"],
    )
    optimization_text = """
    For each model type, we will optimize and save the model according to the number of features selected. 
    While there is no need to optimize the linear regression model, we will optimize the LightGBM model using
    Recursive Feature Elimination (RFE) and GridSearchCV to find the best hyperparameters.
    """
    st.markdown(optimization_text)
    n_features_selected = st.slider(
        "Select the desired number of features",
        min_value=min(model_given_n_features.keys()),
        max_value=max(model_given_n_features.keys()),
        value=default_n_features,
        step=1,
    )
    return n_features_selected


def get_model_optimized(
    model_given_n_features: dict,
    model_type: dict,
    n_features_selected: int,
    X: pd.DataFrame = None,
    y: pd.Series = None,
) -> Pipeline:
    """Check if the model has already been optimized for the selected number of features,

    Parameters:
        model_given_n_features (dict): Dictionary containing pipelines, scores, and selected features.
        model_type (dict): Dictionary containing model type information, including name and scoring method.
        n_features_selected (int): The number of features selected for optimization.
        X (pd.DataFrame): The input features.
        y (pd.Series): The target variable.

    Returns:
        Pipeline: The optimized pipeline for the selected model type and number of features.
    """
    if n_features_selected not in st.session_state.optimized_models[model_type["name"]]:
        if model_type["name"] == "LightGBM":
            pipeline_optimized = optimize_selected_lgbm(X, y, n_features_selected)
        elif model_type["name"] == "Linear Regression":
            pipeline_optimized = model_given_n_features[n_features_selected]["pipeline"]
        else:
            raise ValueError(
                f"Model type {model_type['name']} not supported for optimization."
            )
        st.session_state.optimized_models[model_type["name"]][
            n_features_selected
        ] = pipeline_optimized
    else:
        pipeline_optimized = st.session_state.optimized_models[model_type["name"]][
            n_features_selected
        ]
    return pipeline_optimized


def ui_model_optimization(
    model_given_n_features: dict,
    model_type: dict,
    n_features_selected: int,
    X: pd.DataFrame = None,
    y: pd.Series = None,
) -> Pipeline:
    """User interface for optimizing the selected model based on the number of features.

    Parameters:
        model_given_n_features (dict): Dictionary containing pipelines, scores, and selected features.
        model_type (dict): Dictionary containing model type information, including name and scoring method.
        n_features_selected (int): The number of features selected for optimization.
        X (pd.DataFrame): The input features.
        y (pd.Series): The target variable.

    Returns:
        Pipeline: The optimized pipeline for the selected model type and number of features.
    """
    st.subheader("Summary of Chosen Model")
    pipeline_optimized = get_model_optimized(
        model_given_n_features, model_type, n_features_selected, X, y
    )
    # Get the selected pipeline
    if model_type["name"] == "LightGBM":
        features_df = get_features_from_pipeline(pipeline_optimized, X)
        best_params = pipeline_optimized.named_steps["regression"].get_params()
        best_params = pd.DataFrame.from_dict(
            best_params, orient="index", columns=["Value"]
        ).reset_index()
        AgGrid(best_params, height=300)
    else:
        features_df = model_given_n_features[n_features_selected]["features_df"]
    AgGrid(features_df, height=300)
    return pipeline_optimized


# %%


@st.cache_data
def get_model_train_defaults() -> tuple[dict, dict, dict]:
    train_models_over_range = {}
    optimized_models = {mo: {} for mo in model_type_options.keys()}
    saved_models = {}
    X, y, data_processing, response_feature = initialize_data()

    for model_type_selected in model_type_options.keys():
        # Run training for the selected model type
        model_given_n_features = train_regression_over_range(
            X, y, data_processing, model_type_options[model_type_selected]
        )
        train_models_over_range[model_type_selected] = model_given_n_features

        # Get the number of features with the maximum mean score
        n_features_selected = max(
            model_given_n_features.keys(),
            key=lambda k: model_given_n_features[k]["mean_score"],
        )

        # Save the model with the selected number of features
        if model_type_selected == "LightGBM":
            pipeline_optimized = optimize_selected_lgbm(X, y, n_features_selected)
        elif model_type_selected == "Linear Regression":
            pipeline_optimized = model_given_n_features[n_features_selected]["pipeline"]
        else:
            raise ValueError(
                f"Model type {model_type_selected} not supported for optimization."
            )
        optimized_models[model_type_selected][n_features_selected] = pipeline_optimized
        saved_models[model_type_selected] = pipeline_optimized
    return train_models_over_range, optimized_models, saved_models


def initialize_default_modeling():
    train_models_over_range, optimized_models, saved_models = get_model_train_defaults()
    st.session_state.train_models_over_range = train_models_over_range
    st.session_state.optimized_models = optimized_models
    st.session_state.saved_models = saved_models
    return


def reset_modeling_session_state() -> None:
    st.session_state.train_models_over_range = {}
    st.session_state.optimized_models = {mo: {} for mo in model_type_options.keys()}
    st.session_state.saved_models = {}
    return


def main():
    X, y, data_processing, response_feature = initialize_data()
    st.title("Model Training")
    st.write(
        f"To predict {response_feature}, we will train models over a range of features and optimize them based on the selected model type."
    )
    model_type_selected = st.selectbox(
        "Select Model to Optimize", options=model_type_options.keys(), index=0
    )

    # Run training for the selected model type
    model_given_n_features = get_models_over_range(
        model_type_options[model_type_selected], X, y, data_processing
    )
    ui_model_scores(model_given_n_features, model_type_options[model_type_selected])

    n_features_selected = ui_feature_selection(model_given_n_features)
    model_save = ui_model_optimization(
        model_given_n_features,
        model_type_options[model_type_selected],
        n_features_selected,
        X,
        y,
    )

    save_model_button = st.button("Save Model")
    if save_model_button:
        st.session_state.saved_models[model_type_selected] = model_save
        st.write(f"Model {model_type_selected} saved.")
        st.write(model_save)


if __name__ == "__main__":
    main()
