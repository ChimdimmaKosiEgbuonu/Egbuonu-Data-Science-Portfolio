#importing neccesary libraries

import pandas as pd            
import seaborn as sns          
import matplotlib.pyplot as plt  
import streamlit as st        


#sets title and description of Streamlit app
st.title("Happy Data, Happy Feet: A Streamlit App on the Palmer's Penguins Dataset")
st.write("Description of App: This is my very first Streamlit app. This app allows you to learn more about penguins through interactive features and engaging visualizations. Thank you for checking it out and have fun!")


#calls the .csv dataset file that will be used for the app
df = pd.read_csv("data/penguins.csv")
st.write("This is the data that this app will be analyzing:")

#sets dataframe for the app
st.dataframe(df)

#sets title for summary statistics portion of the app and displays the results in a dataframe
st.write("**Summary Statistics of the Dataset:**")
st.dataframe(df.describe())

#interactive filtering for app
choose_island = st.multiselect(
  "Pick an Island!",
  ["Biscoe", "Dream", "Torgersen"],
  default = "Biscoe"
)

#shows what the user picked
st.write(f"You choose {choose_island}!")
  
  
