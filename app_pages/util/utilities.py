import pandas as pd
import streamlit as st
import yaml
import os
import pickle

# %% Project Configuration


class AppConfig:
    """Configuration for the application.

    Attributes:
        name (str): The name of the application.
        version (str): The version of the application.
    """

    def __init__(self, config: dict):
        """Initialize the application configuration.

        Parameters:
            config (dict): Configuration dictionary containing app settings.
        """
        self.name = config.get("name")
        self.version = config.get("version")


class SnowflakeConfig:
    """Configuration for Snowflake connection.

    Attributes:
        database (str): The name of the Snowflake database.
        schema (str): The name of the Snowflake schema.
    """

    def __init__(self, config: dict):
        """Initialize the Snowflake configuration.

        Parameters:
            config (dict): Configuration dictionary containing Snowflake settings.
        """
        self.database = config.get("database")
        self.schema = config.get("schema")


class PathConfig:
    """Configuration for file paths.

    Attributes:
        data (str): The path to the data directory.
        sql (str): The path to the SQL scripts directory.
    """

    def __init__(self, config: dict):
        """Initialize the path configuration.

        Parameters:
            config (dict): Configuration dictionary containing path settings.
        """
        self.data = config.get("data")
        self.sql = config.get("sql")


class DataProcessingConfig:
    """Configuration for data processing.

    Attributes:
        drop_threshold (float): The percentage threshold for dropping features with missing values.
        data_type_options (list): List of options for data types (e.g., numeric, boolean, categorical).
        missing_handling_options (list): List of options for handling missing values (e.g., "Drop", "Impute", etc.).
    """

    def __init__(self, config: dict):
        """Initialize the data processing configuration.

        Parameters:
            config (dict): Configuration dictionary containing data processing settings.
        """
        self.drop_threshold = config.get("drop_threshold", 0.25)
        self.data_type_options = config.get("data_type_options", [])
        self.missing_handling_options = config.get("missing_handling_options", [])


class ModelingConfig:
    """Configuration for modeling options.

    Attributes:
        model_type_options (dict): Dictionary of model type options with their parameters.
        performance_metrics (dict): Dictionary of performance metrics to evaluate models.
    """

    def __init__(self, config: dict):
        """Initialize the model configuration.

        Parameters:
            config (dict): Configuration dictionary containing model settings.
        """
        self.model_type_options = config.get("model_type_options", {})
        self.performance_metrics = config.get("performance_metrics", {})


class Config:
    """Main configuration class that loads configurations from a YAML file.

    Attributes:
        app (AppConfig): Application configuration.
        snowflake (SnowflakeConfig): Snowflake connection configuration.
        path (PathConfig): Path configuration for file directories.
    """

    def __init__(self, config_file: str = "config.yaml"):
        """Initialize the configuration by loading from a YAML file.

        Parameters:
            config_file (str): The path to the configuration YAML file.
        """
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        self.app = AppConfig(config.get("app", {}))
        self.snowflake = SnowflakeConfig(config.get("snowflake", {}))
        self.path = PathConfig(config.get("path", {}))
        self.data_processing = DataProcessingConfig(config.get("data_processing", {}))
        self.modeling = ModelingConfig(config.get("modeling", {}))


config = Config()

# %% Data Loading and Saving Functions From File


def load_data_from_file(
    file_name: str, data_folder: str = config.path.data
) -> pd.DataFrame:
    """Load data from a file into a DataFrame.

    Parameters:
        file_path (str): The path to the file.
        data_folder (str): The folder where the file is located.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    file_path = os.path.join(data_folder, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_name} does not exist in the data folder {data_folder}."
        )
    elif file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".pkl"):
        with open(file_path, "rb") as file:
            return pickle.load(file)
        return None
    else:
        raise ValueError(
            "Unsupported file format. Only CSV and Pickle files are supported."
        )


def save_to_pickle(object, file_name: str, data_folder: str = config.path.data):
    """Save an object to a pickle file.

    Parameters:
        object: The object to save.
        file_name (str): The name of the file to save the object to.
        data_folder (str): The folder where the file will be saved.
    """
    if not file_name.endswith(".pkl"):
        file_name += ".pkl"
    file_path = os.path.join(data_folder, file_name)
    with open(file_path, "wb") as file:
        pickle.dump(object, file)
    return


# %% Functions for Data Processing


def clean_processed_feature(feature: str) -> str:
    """Clean the processed feature name by removing prefixes output from LR preprocessor.

    Parameters:
        feature (str): The feature name to clean.

    Returns:
        str: The cleaned feature name.
    """
    feature_cleaned = (
        feature.replace("numeric__", "")
        .replace("binary__", "")
        .replace("categorical__", "")
    )
    return feature_cleaned


def clean_feature_name(feature: str) -> str:
    """Clean the feature name for display.

    Parameters:
        feature (str): The feature name to clean.
    Returns:
        str: The cleaned feature name, with underscores replaced by spaces and capitalized.
    """
    return " ".join([f.capitalize() for f in feature.split("_")])


@st.cache_data
def get_default_data_raw() -> pd.DataFrame:
    """Initialize the raw data by loading it from a file.

    Returns:
        pd.DataFrame: The raw data DataFrame.
    """
    speed_dating_data = load_data_from_file(file_name="speed_dating_data_cleaned.pkl")
    # join the partner's reponses
    speed_dating_data = pd.merge(
        speed_dating_data,
        speed_dating_data,
        left_on=["pid", "iid"],
        right_on=["iid", "pid"],
        suffixes=("", "_partner"),
        how="left",
    )

    # Summarize an individual
    # # group by iid and get the first age, gender and income, and the mean decision_partner
    data_raw = (
        speed_dating_data.groupby("iid")
        .agg(
            {
                "decision_partner": "mean",
                "age": "first",
                "gender": "first",
                "college_sat_score": "first",  # SAT score
                "income_zip_code": "first",  # income based on zip code
                "date_frequency": "first",  # how often do you go on dates
                "go_out": "first",  # how often do you go out
                "expect_happy": "first",  # how happy do you expect to be with the people, taken before the event
                "expect_matches": "first",  # how many people do you think you'll be interested in, taken before the event
                "estimated_matches": "first",  # how many matches do you estimate, taken at the end of the night
                "satisfaction": "first",  # how satisfied were you with the people you met, taken the day after
                # what you look for in a partner
                "attractive_looking": "first",
                "ambitious_looking": "first",
                "fun_looking": "first",
                "intelligent_looking": "first",
                "shared_interests_looking": "first",
                "sincere_looking": "first",
                # what you think your same gender looks for
                "attractive_same": "first",
                "ambitious_same": "first",
                "fun_same": "first",
                "intelligent_same": "first",
                "shared_interests_same": "first",
                "sincere_same": "first",
                # what you think the opposite gender looks for
                "attractive_opposite": "first",
                "ambitious_opposite": "first",
                "fun_opposite": "first",
                "intelligent_opposite": "first",
                "shared_interests_opposite": "first",
                "sincere_opposite": "first",
                # how do you rate yourself
                "attractive_yourself": "first",
                "ambitious_yourself": "first",
                "fun_yourself": "first",
                "intelligent_yourself": "first",
                "sincere_yourself": "first",
                # how do you think others rate you
                "attractive_others": "first",
                "ambitious_others": "first",
                "fun_others": "first",
                "intelligent_others": "first",
                "sincere_others": "first",
                # how did your partner rate you
                "attractive_partner": "mean",
                "ambitious_partner": "mean",
                "fun_partner": "mean",
                "intelligent_partner": "mean",
                "shared_interests_partner": "mean",
                "sincere_partner": "mean",
            }
        )
        .reset_index()
    )
    data_raw = data_raw.rename(columns={"decision_partner": "match_rate"})
    data_raw["match_rate"] = pd.to_numeric(data_raw["match_rate"], errors="coerce")
    data_raw = data_raw.drop(columns=["iid"])  # drop iid column

    # clean column names
    data_raw.columns = [clean_feature_name(col) for col in data_raw.columns]
    return data_raw


@st.cache_data
def get_default_data_processing(
    data_raw: pd.DataFrame,
    drop_threshold: int = config.data_processing.drop_threshold,
    data_type_options: list = config.data_processing.data_type_options,
    missing_handling_options: list = config.data_processing.missing_handling_options,
) -> dict:
    """Initialize data processing options based on the raw data.

    Parameters:
        data_raw (pd.DataFrame): The raw data to process.
        drop_threshold (int): The percentage threshold for dropping features with missing values.

    Returns:
        dict: A dictionary containing data processing options for each feature.
    """

    drop_list = [" Same", " Opposite", "Yourself", " Others"]

    data_processing = {}
    for f in data_raw.columns:

        # drop feature if it has too many missing values or contains certain keywords
        drop_feature_flag = data_raw[f].isnull().mean() * 100 >= drop_threshold or any(
            d in f for d in drop_list
        )

        # if data type is boolean
        if pd.api.types.is_bool_dtype(data_raw[f]):
            data_processing[f] = {
                "data_type": (
                    data_type_options[1] if not drop_feature_flag else "Drop Feature"
                ),
                "missing_handling": missing_handling_options[0],
                "outlier_min": 0,
                "outlier_max": 1,
            }
        # if data type is numeric
        elif pd.api.types.is_numeric_dtype(data_raw[f]):
            data_processing[f] = {
                "data_type": (
                    data_type_options[0] if not drop_feature_flag else "Drop Feature"
                ),
                "missing_handling": missing_handling_options[0],
                "outlier_min": data_raw[f].min(),
                "outlier_max": data_raw[f].max(),
            }
        # if data type is categorical
        elif pd.api.types.is_categorical_dtype(data_raw[f]):
            data_processing[f] = {
                "data_type": (
                    data_type_options[2] if not drop_feature_flag else "Drop Feature"
                ),
                "missing_handling": missing_handling_options[0],
                "outlier_min": data_raw[f].min(),
                "outlier_max": data_raw[f].max(),
            }

    return data_processing


def apply_data_processing(
    data_raw: pd.DataFrame, data_processing: dict
) -> tuple[pd.DataFrame, dict]:
    """Apply data processing options to the raw data.

    Parameters:
        data_raw (pd.DataFrame): The raw data to process.
        data_processing (dict): A dictionary containing data processing options for each feature.

    Returns:
        pd.DataFrame: The processed data after applying the data processing options.
        dict: The updated data processing options with the number of dropped rows for each feature.
    """
    data = data_raw.copy()
    dropped_rows = {feature: 0 for feature in data_raw.columns}
    for feature in data_raw.columns:
        if feature not in data_processing:
            dropped_rows[feature] = 0

        elif data_processing[feature]["data_type"] == "Drop Feature":
            data = data.drop(columns=[feature])

        else:
            n_rows = data.shape[0]
            # handle missing values
            if data_processing[feature]["missing_handling"] == "Drop":
                # drop rows with missing values in this feature
                data = data[data[feature].notnull()]
            # handle outliers
            if data_processing[feature]["data_type"] == "Numeric":
                data = data[
                    (data[feature] >= data_processing[feature]["outlier_min"])
                    & (data[feature] <= data_processing[feature]["outlier_max"])
                ]
            dropped_rows[feature] = n_rows - data.shape[0]

    return data, dropped_rows


# %% Session State Initialization


def initialize_data():
    data_processing = st.session_state.data_processing
    response_feature = st.session_state.response_feature
    data = st.session_state.data
    X = data.drop(columns=[response_feature])
    y = data[response_feature]

    return X, y, data_processing, response_feature


def initialize_default_data_processing():
    """Initialize session state for data processing and raw data.

    Returns:
        pd.DataFrame: The raw data after processing, excluding certain columns.
    """
    data_raw = get_default_data_raw()
    data_processing = get_default_data_processing(data_raw)
    data, dropped_rows = apply_data_processing(data_raw, data_processing)
    if "data_processing" not in st.session_state:
        st.session_state.data_processing = data_processing
    if "data" not in st.session_state:
        st.session_state.data = data
    if "response_feature" not in st.session_state:
        # default to first feature
        st.session_state.response_feature = data_raw.columns[0]

    return
