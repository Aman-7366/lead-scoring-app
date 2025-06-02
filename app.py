import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
with open("/mnt/data/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("/mnt/data/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("/mnt/data/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)


st.header('Real estate Lead Scoring App')

source = st.selectbox("Source", label_encoders['Source'].classes_)
profession = st.selectbox("Profession", label_encoders['Profession'].classes_)
property_type = st.selectbox("Property Type", label_encoders['Property_Type'].classes_)
age = st.slider("Age", 18, 70, 30)
budget = st.number_input("Budget (in lakhs)", min_value=10, max_value=1000, step=10)
interactions = st.slider("Number of Interactions", 0, 50, 3)
days_since_contact = st.slider("Days Since Last Contact", 0, 365, 7)

source_encoded = label_encoders['Source'].transform([source])[0]
profession_encoded = label_encoders['Profession'].transform([profession])[0]
property_encoded = label_encoders['Property_Type'].transform([property_type])[0]

input_df = pd.DataFrame([[source_encoded, age, profession_encoded, property_encoded,
                          budget, interactions, days_since_contact]],
                        columns=["Source", "Age", "Profession", "Property_Type",
                                 "Budget", "Interactions", "Days_Since_Contact"])

scaled_input = scaler.transform(input_df)

# Predict
if st.button("Predict Conversion Probability"):
    prediction = model.predict_proba(scaled_input)[0][1]
    st.success(f"🔮 Conversion Probability: {round(prediction * 100, 2)}%")
