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

from eda_analysis import clean_feature_name

# %% Helper Functions

def clean_processed_feature(feature: str) -> str:
    feature_cleaned = (
        feature.replace("numeric__", "")
        .replace("binary__", "")
        .replace("categorical__", "")
    )
    return feature_cleaned

def get_features_from_pipeline(pipeline, X, clean_flag=True):
    features_df = pd.DataFrame()
    # st.write(pipeline)
    # lightgbm grid search
    if "regression" not in pipeline.named_steps and "lgbm_grid" in pipeline.named_steps:
        selected_features = X.columns[
            pipeline.named_steps["feature_selection"].get_support()
        ]
        if clean_flag:
            selected_features = [clean_feature_name(f) for f in selected_features]
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
    elif "feature_selection" in pipeline.named_steps and hasattr(pipeline.named_steps["regression"], "feature_importances_"):
        selected_features = X.columns[
            pipeline.named_steps["feature_selection"].get_support()
        ]
        if clean_flag:
            selected_features = [clean_feature_name(f) for f in selected_features]
        feature_importances = pipeline.named_steps[
            "regression"
        ].feature_importances_
        sorted_index = np.argsort(feature_importances)[::-1]
        feature_importances = feature_importances[sorted_index]
        selected_features = np.array(selected_features)[sorted_index]
        features_df = pd.DataFrame(
            {"Feature": selected_features, "Feature Importance": feature_importances}
        )
    # linear regression
    elif "preprocessor" in pipeline.named_steps and hasattr(pipeline.named_steps["regression"], "coef_"):
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
        if clean_flag:
            selected_features = [clean_feature_name(f) for f in selected_features]
        sorted_index = np.argsort(np.abs(coefficients))[::-1] # sort by decreasing feature importance based on coef
        coefficients = coefficients[sorted_index]
        selected_features = np.array(selected_features)[sorted_index]
        features_df = pd.DataFrame(
            {"Feature": selected_features, "Coefficient": coefficients}
        )
    return features_df

# %% Setup the data

data = st.session_state.data
response_feature = st.session_state.response_feature
response_feature_cleaned = clean_feature_name(response_feature)
data_processing = st.session_state.data_processing
X = data.drop(columns=[response_feature])
y = data[response_feature]

if "saved_models" not in st.session_state:
    st.session_state.saved_models = {}

# %% User Interaction

st.title(f"Predict {response_feature_cleaned}")
model_options = ["Linear Regression", "LightGBM"]
model_type_selected = st.selectbox(
    "Select Model to Optimize", options=model_options, index=0
)
st.write(f"Saved models: {list(st.session_state.saved_models.keys())}")

# %% Combined Pipeline Functions

scoring_map = {
    "Linear Regression": {"scoring": "r2", "name": "R2"},
    "LightGBM": {"scoring": "neg_mean_squared_error", "name": "Neg-MSE"}
}

@st.cache_data
def train_regression_over_range(X, y, data_processing, model_type_selected):
    #identify feature types
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
    if model_type_selected == "Linear Regression":
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

    n_features_range = np.arange(
        1, len(X.columns) + 1
    )  # Number of features to select for RFE
    estimator = LinearRegression() if model_type_selected == "Linear Regression" else LGBMRegressor(verbose=-1)
    scoring = scoring_map[model_type_selected]["scoring"]
    rkf = RepeatedKFold(n_splits=2, n_repeats=3, random_state=42)
    model_given_n_features = {}
    for n_features in n_features_range:
        rfe = RFE(estimator, n_features_to_select=n_features)
        model_given_n_features[n_features] = {'pipeline': Pipeline(
            steps=steps + [("feature_selection", rfe), ("regression", estimator)]
        )}
        # pipeline = Pipeline(
        #     steps=steps + [("feature_selection", rfe), ("regression", estimator)]
        # )
        scores = cross_val_score(model_given_n_features[n_features]["pipeline"], X, y, scoring=scoring, cv=rkf, n_jobs=-1)
        model_given_n_features[n_features]["mean_score"] = np.mean(scores)
        # get linear regression features and importances
        model_given_n_features[n_features]["pipeline"].fit(X, y)
        model_given_n_features[n_features]["features_df"] = get_features_from_pipeline(model_given_n_features[n_features]["pipeline"], X)
        # model_given_n_features[n_features] = {
        #     "pipeline": pipeline,
        #     "mean_score": np.mean(scores),
        #     "features_df": features_df
        # }

    return n_features_range, model_given_n_features

@st.cache_data
def optimize_selected_lgbm(X, y, n_features_selected):
    # build new pipeline with selected features and grid search
    rfe = RFE(LGBMRegressor(verbose=-1), n_features_to_select=n_features_selected)
    lgbm_grid = {
        "num_leaves": [5, 20, 31],
        "learning_rate": [0.05, 0.1, 0.2],
        "n_estimators": [50, 100, 150],
        "max_depth": [2, 4, 6, 8],  # range(2, 10, 2)
    }
    grid_search = GridSearchCV(
        LGBMRegressor(verbose=-1), lgbm_grid, cv=3, n_jobs=-1, scoring="neg_mean_squared_error"
    )
    pipeline = Pipeline(steps=[("feature_selection", rfe), ("lgbm_grid", grid_search)])
    pipeline.fit(X, y)

    return pipeline

def regression_model(X, y, data_processing, model_type_selected):
    st.subheader(f"{model_type_selected} Pipeline")
    n_features_range, model_given_n_features = train_regression_over_range(X, y, data_processing, model_type_selected)
    
    # Plot the scores for different numbers of features
    scores = [model_given_n_features[n]["mean_score"] for n in n_features_range]
    score_name = scoring_map[model_type_selected]["name"]
    st.write(f"Mean {score_name} Scores for Different Numbers of Features Selected (using RFE)")
    fig_lr, ax_lr = plt.subplots()
    ax_lr = sns.lineplot(x=n_features_range, y=scores, marker="o")
    plt.xlabel("Number of Features Selected")
    plt.ylabel(f"Mean {score_name}")
    plt.grid()
    st.pyplot(fig_lr)

    # Display the scores and selected features in a table
    selected_features_list = [model_given_n_features[n]["features_df"]["Feature"].tolist() for n in n_features_range]
    models_summary = pd.DataFrame(
        {
            "Number of Features": n_features_range,
            f"Mean {score_name}": scores,
            "Selected Features": selected_features_list,
        }
    )
    AgGrid(models_summary, height=300)

    # Optimization based on user-selected number of features
    st.subheader("Select Number of Features for Optimization")
    default_n_features = np.argmax(scores) + 1  # get the index with maximum r2
    n_features_selected = st.slider(
        "Select Number of Features Corresponding to Best Performance",
        min_value=1,
        max_value=len(X.columns)-1,
        value=default_n_features,
        step=1,
    )

    st.subheader("Summary of Chosen Model")
    
    # Get the selected pipeline
    if model_type_selected == "LightGBM":
        start_time = time.time()
        pipeline = optimize_selected_lgbm(X, y, n_features_selected)
        st.write(f"Optimization completed in {time.time() - start_time:.2f} seconds")
        features_df = get_features_from_pipeline(pipeline, X)
        AgGrid(features_df, height=300)
        best_params = pipeline.named_steps["lgbm_grid"].best_params_
        st.write("Best Parameters:", best_params)
        st.write(f"Best {score_name}: {pipeline.named_steps['lgbm_grid'].best_score_:.4f}")
        best_params["verbose"] = -1
        # use the best parameters to build the final pipeline
        lgbm_optimized = LGBMRegressor(**best_params)
        pipeline_final = Pipeline(
            steps=[("feature_selection", pipeline.named_steps["feature_selection"]), ("regression", lgbm_optimized)]
        )
        pipeline_final.fit(X, y)
        return pipeline_final
    else:
        AgGrid(model_given_n_features[n_features_selected]["features_df"], height=300)
        return model_given_n_features[n_features_selected]["pipeline"]

# %% User Interaction for Model Selection and Training

model_save = regression_model(X, y, data_processing, model_type_selected)

save_model_button = st.button("Save Model")
if save_model_button:
    st.session_state.saved_models[model_type_selected] = model_save
    st.write(f"Model {model_type_selected} saved.")
    st.write(model_save)
