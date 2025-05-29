import streamlit as st

import joblib 
import numpy as np

# load your trained model and vectorizer

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl") 

st.set_page_config(page_title="toxic comment detector", layout="centered")

# title and description

st.write("Enter the Comment below and the model will tell you if it's Toxic or Not")

#user input

user_input = st.text_area("Enter a Comment")

if st.button("Analyze"): 
    if user_input.strip() =="":
        st.warning("please enter a comment")
    else:
        #transform and predict

        input_vector = vectorizer.transform([user_input]) 
        prediction = model.predict(input_vector)[0]
        prob = model.predict_proba(input_vector)[0] [1]

        #probablity of being toxic

        if prediction == 1:
            st.error(f" toxic comment detector ({prob:.2%} confidence)")
        else:
            st.success(f" clean comment ({1 - prob:.2%} confidence)")