import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv(r"F:\Pima Indians Diabetes\diabetes.csv")

# Train model
X = df.drop('Outcome', axis=1)
y = df['Outcome']
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit UI
st.title("🩺 Diabetes Prediction App")
st.write("Enter the values below to predict the chances of having diabetes.")

# User Inputs
preg = st.number_input('Pregnancies', 0, 20)
glucose = st.slider('Glucose', 0, 200, 110)
bp = st.slider('Blood Pressure', 0, 140, 70)
skin = st.slider('Skin Thickness', 0, 100, 20)
insulin = st.slider('Insulin', 0, 900, 80)
bmi = st.slider('BMI', 0.0, 70.0, 25.0)
dpf = st.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5)
age = st.slider('Age', 10, 100, 30)

# Predict button
if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("You may have Diabetes. Please consult a doctor.")
    else:
        st.success("You are likely not diabetic.")
