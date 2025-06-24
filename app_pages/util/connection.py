"""Helper Functions

This module contains helper functions for interacting with Snowflake, loading data, and managing
configurations.
"""

# %% Libraries

import os
import pandas as pd
from snowflake import connector as sfc
import streamlit as st

from app_pages.util.utilities import config

# %% Snowflake Base Functions


@st.cache_resource
def _initialize_snowflake_connection() -> sfc.SnowflakeConnection:
    """Initialize a Snowflake connection using credentials from Streamlit secrets.

    Returns:
        sfc.SnowflakeConnection: A Snowflake connection object.
    """
    return sfc.connect(
        **st.secrets["snowflake"],
        database=config.snowflake.database.upper(),
        schema=config.snowflake.schema.upper(),
        client_session_keep_alive=True,
    )


_session = _initialize_snowflake_connection()


def _write_to_snowflake(data: pd.DataFrame, table_name: str):
    """Write a DataFrame to a Snowflake table.

    Parameters:
        data (pd.DataFrame): The DataFrame to write.
        table_name (str): The name of the table to write to.
    """
    data.columns = data.columns.str.upper()  # Convert column names to uppercase
    success, num_chunks, num_rows, _ = sfc.pandas_tools.write_pandas(
        _session, data, table_name=table_name.upper(), overwrite=True
    )
    if not success:
        raise Exception(f"Failed to write data to Snowflake table {table_name}.")
    return


def _read_from_snowflake(sql_query: str) -> pd.DataFrame:
    """Read data from Snowflake using a SQL query.

    Parameters:
        sql_query (str): The SQL query to execute.
    Returns:
        pd.DataFrame: The data returned by the query.
    """
    data = _session.cursor().execute(sql_query).fetch_pandas_all()
    data.columns = data.columns.str.lower()
    return data


def _fix_data_types(
    data: pd.DataFrame, data_types: dict, remove_commas_columns: list = []
) -> pd.DataFrame:
    """Fix data types of a DataFrame and remove commas from specified columns.

    Parameters:
        data (pd.DataFrame): The DataFrame to process.
        data_types (dict): A dictionary mapping column names to their desired data types.
        remove_commas_columns (list): List of columns from which to remove commas and ".00".
    Returns:
        pd.DataFrame: The processed DataFrame with fixed data types and cleaned columns.
    """
    # automatic data type conversion helps some
    data = data.convert_dtypes()

    # remove commas and .00 from specific columns
    for col in remove_commas_columns:
        if col in data.columns:
            data[col] = data[col].str.replace(",", "")
            data[col] = data[col].str.replace(".00", "")

    # convert data types
    data = data.astype(data_types)

    return data


# %% Call Functions


def initialize_snowflake_table(
    file_name: str,
    table_name: str,
    data_folder: str = config.path.data,
    encoding: str = None,
    data_types: dict = {},
    remove_commas_columns: list = [],
) -> pd.DataFrame:
    """Initialize a Snowflake table by reading data from a CSV file and writing it to Snowflake.
    Parameters:
        file_name (str): The name of the CSV file to read.
        table_name (str): The name of the Snowflake table to create or overwrite.
        data_folder (str): The folder where the CSV file is located.
        encoding (str): The encoding of the CSV file (default is None, which uses UTF-8).
        data_types (dict): A dictionary mapping column names to their desired data types.
        remove_commas_columns (list): List of columns from which to remove commas and ".00".
    Returns:
        pd.DataFrame: The DataFrame read from the CSV file and written to Snowflake.
    """
    if file_name.endswith(".csv"):
        data = pd.read_csv(os.path.join(data_folder, file_name), encoding=encoding)
        if data_types:
            data = _fix_data_types(data, data_types, remove_commas_columns)
        _write_to_snowflake(data, table_name=table_name)
    else:
        raise ValueError("Unsupported file format. Only CSV files are supported.")
    return data


@st.cache_data
def query_snowflake(
    query_input: str, sql_folder: str = config.path.sql, data_types: dict = {}
) -> pd.DataFrame:
    """Query Snowflake using a SQL query or a file.

    Parameters:
        query_input (str): The SQL query string or the name of a SQL file (with .sql extension).
        sql_folder (str): The folder where SQL files are located.
        data_types (dict): A dictionary mapping column names to their desired data types.
    Returns:
        pd.DataFrame: The data returned by the query.
    """
    if query_input.endswith(".sql"):
        query_file = os.path.join(sql_folder, query_input)
        with open(query_file, "r") as file:
            sql_query = file.read()
    elif "select" in query_input.lower():
        sql_query = query_input
    else:
        sql_query = f"SELECT * FROM {query_input.upper()}"
    data = _read_from_snowflake(sql_query)
    if data_types:
        data = _fix_data_types(data, data_types)
    return data


def drop_snowflake_table(table_name: str):
    """Drop a Snowflake table.

    Parameters:
        table_name (str): The name of the table to drop.
    """
    sql_query = f"DROP TABLE IF EXISTS {table_name.upper()}"
    _session.cursor().execute(sql_query)
    return



