
# %% Libraries

import os
import pandas as pd
import snowflake.connector
import snowflake.connector.pandas_tools
import streamlit as st
import yaml
from ydata_profiling import ProfileReport
import pickle

# %% Load Configuration

class AppConfig:
    def __init__(self, config: dict):
        self.name = config.get('name')
        self.version = config.get('version')

class SnowflakeConfig:
    def __init__(self, config: dict):
        self.database = config.get('database')
        self.schema = config.get('schema')

class PathConfig:
    def __init__(self, config: dict):
        self.data = config.get('data')
        self.sql = config.get('sql')

class Config:
    def __init__(self, config_file: str = 'config.yaml'):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        self.app = AppConfig(config.get('app', {}))
        self.snowflake = SnowflakeConfig(config.get('snowflake', {}))
        self.path = PathConfig(config.get('path', {}))

config = Config()

# %% Snowflake Base Functions

@st.cache_resource
def _initialize_snowflake_connection() -> snowflake.connector.SnowflakeConnection:
    return snowflake.connector.connect(
        **st.secrets['snowflake'],
        database=config.snowflake.database.upper(),
        schema=config.snowflake.schema.upper(),
        client_session_keep_alive=True
    )

_session = _initialize_snowflake_connection()

def _write_to_snowflake(data: pd.DataFrame, table_name: str):
    """
    Write a DataFrame to a Snowflake table.

    Parameters:
    data (pd.DataFrame): The DataFrame to write.
    table_name (str): The name of the table to write to.
    """
    data.columns = data.columns.str.upper()  # Convert column names to uppercase
    success, num_chunks, num_rows, _ = snowflake.connector.pandas_tools.write_pandas(
        _session,
        data,
        table_name=table_name.upper(),
        overwrite=True
    )
    if not success:
        raise Exception(f"Failed to write data to Snowflake table {table_name}.")
    return

def _read_from_snowflake(sql_query: str) -> pd.DataFrame:
    data = _session.cursor().execute(sql_query).fetch_pandas_all()
    data.columns = data.columns.str.lower()
    return data

def _fix_data_types(data: pd.DataFrame, data_types: dict, remove_commas_columns: list=[]) -> pd.DataFrame:
    # automatic data type conversion helps some
    data = data.convert_dtypes()

    # remove commas and .00 from specific columns
    for col in remove_commas_columns:
        if col in data.columns:
            data[col] = data[col].str.replace(',', '')
            data[col] = data[col].str.replace('.00', '')

    # convert data types 
    data = data.astype(data_types)

    return data

# %% Call Functions

def initialize_snowflake_table(
        file_name: str, 
        table_name: str, 
        data_folder: str=config.path.data, 
        encoding: str=None,
        data_types: dict={},
        remove_commas_columns: list=[],
        ) -> pd.DataFrame:
    if file_name.endswith('.csv'):
        data = pd.read_csv(os.path.join(data_folder, file_name), encoding=encoding)
        if data_types:
            data = _fix_data_types(data, data_types, remove_commas_columns)
        _write_to_snowflake(
            data,
            table_name=table_name
        )
    else:
        raise ValueError("Unsupported file format. Only CSV files are supported.")
    return data

@st.cache_data
def query_snowflake(
        query_input: str,
        sql_folder: str=config.path.sql,
        data_types: dict={}) -> pd.DataFrame:
    if query_input.endswith('.sql'):
        query_file = os.path.join(sql_folder, query_input)
        with open(query_file, 'r') as file:
            sql_query = file.read()
    elif 'select' in query_input.lower():
        sql_query = query_input
    else:
        sql_query = f"SELECT * FROM {query_input.upper()}"
    data =_read_from_snowflake(sql_query)
    if data_types:
        data = _fix_data_types(data, data_types)
    return data

def drop_snowflake_table(table_name: str):
    """
    Drop a Snowflake table.

    Parameters:
    table_name (str): The name of the table to drop.
    """
    sql_query = f"DROP TABLE IF EXISTS {table_name.upper()}"
    _session.cursor().execute(sql_query)
    return

# @st.cache_resource
# def create_data_profile(data: pd.DataFrame) -> ProfileReport:
#     return ProfileReport(data, explorative=True)

# @st.cache_data
def load_data_from_file(file_name: str, data_folder: str=config.path.data) -> pd.DataFrame:
    """
    Load data from a file into a DataFrame.

    Parameters:
    file_path (str): The path to the file.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    file_path = os.path.join(data_folder, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_name} does not exist in the data folder {data_folder}.")
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
        return None
    else:
        raise ValueError("Unsupported file format. Only CSV and Pickle files are supported.")

def save_to_pickle(object, file_name: str, data_folder: str=config.path.data):
    """
    Save an object to a pickle file.

    Parameters:
    object: The object to save.
    file_name (str): The name of the file to save the object to.
    data_folder (str): The folder where the file will be saved.
    """
    if not file_name.endswith('.pkl'):
        file_name += '.pkl'
    file_path = os.path.join(data_folder, file_name)
    with open(file_path, 'wb') as file:
        pickle.dump(object, file)
    return


# %%
