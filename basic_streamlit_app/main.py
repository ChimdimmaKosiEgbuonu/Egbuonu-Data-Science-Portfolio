import pandas as pd            
import seaborn as sns          
import matplotlib.pyplot as plt  
import streamlit as st        


st.title("Happy Data, Happy Feet: A Streamlit App on the Palmer's Penguins Dataset")
st.write("This is my first Streamlit app. This app allows you to learn more about penguins through interactive features and engaging visualizations. Thank you for checking it out and have fun!")


df = pd.read_csv("basic_streamlit_app/data/penguins.csv")
st.write("This is the data that this app will be analyzing")
st.dataframe(df)

st.write("**Summary Statistics of the Dataset**")
st.dataframe(df.describe())
