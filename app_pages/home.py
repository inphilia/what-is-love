"""
This is the home page of the Speed Dating Data Analysis app.
"""

import streamlit as st

st.header("Speed Dating Data Analysis")
st.write("Dr. Alan Y Nam")

st.subheader("Introduction")
introduction_text = """
Welcome to my Speed Dating Data Analysis app! This is a fun side project to learn Streamlit and visualization libraries.
There's also a lot of work behind the scenes in setting up the project structure, data processing pipelines, and modeling pipelines.
So while it may look simple, I'm hoping to scale this across multiple projects in the future.

To use this app, navigate through the pages in the order presented in the sidebar.  
-- Exploratory Data Analysis: Understand the data and its features.  
-- Data Preprocessing: Set up how the data is processed for modeling.  
-- Modeling (**Required**): Train and save different models to predict a response feature.  
-- Model Comparison: Compare the performance and feature importance of saved models.  

To get more background on the data, check out the [original dataset and description](https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment/data?select=Speed+Dating+Data.csv).
Or read the [paper](https://business.columbia.edu/sites/default/files-efs/pubfiles/867/fisman%20iyengar.pdf) that describes the data collection and some analysis.
Also check out this awesome [analysis](https://www.kaggle.com/code/jph84562/the-ugly-truth-of-people-decisions-in-speed-dating) to which I owe a lot of inspiration, especially for the EDA section.
"""
st.markdown(introduction_text)

st.subheader("Results")
results_text = """
Unsurprisingly, the most important feature for predicting match rate is attractiveness (as rated by partners), followed by fun, and shared interests.
Gender also plays a role, with women generally having a higher match rate than men.
When it comes to modeling, the linear regression model actually outperforms LightGBM, which is a good reminder that simpler models can often be more effective, especially with smaller datasets.
The performance could likely be improved with better feature engineering and hyperparameter tuning.
"""
st.markdown(results_text)

st.subheader("About Me")
about_me_text = """
I'm a doctor (in nuclear engineering) and therefore fully qualified to give dating advice.

For real, I'm a data scientist with a decade of experience across multiple domains, including engineering, finance, marketing, oversight, and real estate.
You can find me on [LinkedIn](https://www.linkedin.com/in/alan-nam-639a03134/).
"""
st.markdown(about_me_text)
