"""
This script sets up the data for the app to run smoothly.

It reads from a CSV file, initializes a Snowflake table, and saves the data as a pickle file.
This is a one-time setup script that prepares the data for the rest of the app.
It may seem unnecessary, but it served as a good exercise in setting up the project structure,
data processing pipelines, and setting up the Snowflake connection.
"""

# %% Library imports

# import os
# import pandas as pd
from app_pages.util.connection import query_snowflake, save_to_pickle

# %% Initialize Snowflake Table

# read data from CSV
# file_name='Speed Dating Data.csv'
# data_folder = 'data'
# encoding = "ISO-8859-1"
# speed_dating_data = pd.read_csv(os.path.join(data_folder, file_name), encoding=encoding)

# table_name = 'SPEED_DATING'

# drop_snowflake_table(table_name)
# speed_dating_data = initialize_snowflake_table(
#     file_name='Speed Dating Data.csv',
#     table_name=table_name,
#     encoding="ISO-8859-1",
#     data_types = {
#         'gender': 'bool',
#         'match_flag': 'bool',
#         'samerace': 'bool',
#         'dec_o': 'bool',
#         'mn_sat': 'Int16',
#         'tuition': 'Int64',
#         'zipcode': 'Int64',
#         'income': 'Int64'
#     },
#     remove_commas_columns = ['mn_sat', 'tuition', 'zipcode', 'income']
# )

# %% Load the Data from Snowflake

# Query Snowflake for the data
data_types = {"goal": "Int16", "decision": "bool", "pid": "Int16"}
speed_dating_data = query_snowflake("speed_dating.sql", data_types=data_types)

# %% Fix column names

speed_dating_data = speed_dating_data.rename(
    columns={
        "mn_sat": "college_sat_score",
        "income": "income_zip_code",
        "exphappy": "expect_happy",
        "expnum": "expect_matches",
        "match_es": "estimated_matches",
        "satis_2": "satisfaction",
    }
)

attribute_columns = {
    "attr": "attractive",
    "amb": "ambitious",
    "fun": "fun",
    "intel": "intelligent",
    "shar": "shared_interests",
    "sinc": "sincere",
}
attribute_suffixes = {
    "1_1": "_looking",
    "4_1": "_same",
    "2_1": "_opposite",
    "3_1": "_yourself",
    "5_1": "_others",
    "": "",
}
attribute_rename = {
    f"{attr}{suffix}": f"{attribute_columns[attr]}{attribute_suffixes[suffix]}"
    for attr in attribute_columns
    for suffix in attribute_suffixes
}
speed_dating_data = speed_dating_data.rename(columns=attribute_rename)

# %% Save the Data to files

speed_dating_data.to_csv("data/speed_dating_data_cleaned.csv", index=False)
save_to_pickle(speed_dating_data, file_name="speed_dating_data_cleaned.pkl")
