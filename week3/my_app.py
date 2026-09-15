
import streamlit as st

st.title("Hello, streamlit")
st.write("This is my first Streamlit app.")

if st.button("Click Me!"):
    st.write("You clicked the button! Nice work!")
else:
    st.write("Click the button to see what happens!")

import pandas as pd

st.subheader("Exploring Our Dataset")

df = pd.read_csv("data/sample_data.csv")

st.write("Here's our data")
st.dataframe(df)


