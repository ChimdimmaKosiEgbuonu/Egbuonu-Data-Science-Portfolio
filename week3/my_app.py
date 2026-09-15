
import streamlit as st

st.title("Hello, streamlit")
st.write("This is my first Streamlit app.")

if st.button("Click Me!"):
    st.write("You clicked the button! Nice work!")
else:
    st.write("Click the button to see what happens!")

import pandas as pd

st.subheader("Exploring Our Dataset")

df = pd.read_csv("/Users/chimdimmaegbuonu/Documents/Egbuonu-Data-Science-Portfolio/week3/sample_data.csv")