# %% Library imports

# from helper import initialize_snowflake_table
import os
import pandas as pd
from ydata_profiling import ProfileReport
from helper import drop_snowflake_table, initialize_snowflake_table, query_snowflake, load_data_from_file, save_to_pickle

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
data_types = {
    'goal': 'Int16',
    'decision': 'bool',
    'pid': 'Int16'
}
speed_dating_data = query_snowflake('speed_dating.sql', data_types=data_types)

# %% Fix column names

speed_dating_data = speed_dating_data.rename(columns={
    'mn_sat': 'college_sat_score',
    'income': 'income_zip_code',
    'exphappy': 'expect_happy',
    'expnum': 'expect_matches',
    'match_es': 'estimated_matches',
    'satis_2': 'satisfaction',
})

attribute_columns = {
    'attr': 'attractive',
    'amb': 'ambitious',
    'fun': 'fun',
    'intel': 'intelligent',
    'shar': 'shared_interests',
    'sinc': 'sincere'
}
attribute_suffixes = {
    '1_1': '_looking',
    '4_1': '_same',
    '2_1': '_opposite',
    '3_1': '_yourself',
    '5_1': '_others',
    '': ''
}
attribute_rename = {
    f'{attr}{suffix}': f'{attribute_columns[attr]}{attribute_suffixes[suffix]}'
    for attr in attribute_columns
    for suffix in attribute_suffixes
}
speed_dating_data = speed_dating_data.rename(columns=attribute_rename)

# %% Save the Data to files

speed_dating_data.to_csv('data/speed_dating_data_cleaned.csv', index=False)
save_to_pickle(speed_dating_data, file_name='speed_dating_data_cleaned.pkl')

# with open('data/speed_dating_data_cleaned.pkl', 'wb') as file:
#     pickle.dump(speed_dating_data, file)
# speed_dating_data.to_pickle('data/speed_dating_data_cleaned.pkl')

# %% Load the Data from files

speed_dating_data = load_data_from_file(file_name='speed_dating_data_cleaned.pkl')

# %% Save the profile report

profile = ProfileReport(speed_dating_data, explorative=True)
save_to_pickle(profile, file_name='speed_dating_profile_report.pkl')

# %% 

# remove_commas_columns = ['mn_sat', 'tuition', 'zipcode', 'income']


# # automatic data type conversion helps some
# speed_dating_data = speed_dating_data.convert_dtypes()

# # remove commas and .00 from specific columns
# for col in remove_commas_columns:
#     if col in speed_dating_data.columns:
#         print(f'Removing commas from column: {col}')
#         speed_dating_data[col] = speed_dating_data[col].str.replace(',', '')
#         speed_dating_data[col] = speed_dating_data[col].str.replace('.00', '')

# # convert data types 
# speed_dating_data = speed_dating_data.astype(data_types)


# %%
