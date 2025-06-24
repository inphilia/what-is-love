# %% Import Libraries

import streamlit as st

# %% Setup the data

# saved_models = st.session_state.saved_models
# for model_name, model in saved_models.items():
#     st.subheader(f"Model: {model_name}")
#     st.write(model)


# %% Is session state a reference or a copy

if "test_value" not in st.session_state:
    st.session_state.test_value = 0

st.write(f"Test value: {st.session_state.test_value}")
test_value = st.session_state.test_value
test_value += 1

st.write(f"Updated test value: {test_value}")
st.write(f"Session state test value: {st.session_state.test_value}")
