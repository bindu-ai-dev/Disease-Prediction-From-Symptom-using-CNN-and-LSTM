import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load Model and Preprocessing
model = load_model("disease_cnn_lstm_model.h5")
tokenizer = joblib.load("symptom_tokenizer.pkl")
label_encoder = joblib.load("disease_label_encoder.pkl")


st.title("Disease Prediction System")

st.write("Enter symptoms to predict the disease")


# User Input
symptoms = st.text_input("Enter Symptoms")


if st.button("Predict"):

    seq = tokenizer.texts_to_sequences([symptoms])

    seq = pad_sequences(seq, maxlen=30)

    prediction = model.predict(seq)

    pred_class = np.argmax(prediction)

    disease = label_encoder.inverse_transform([pred_class])

    confidence = np.max(prediction)

    st.success(f"Predicted Disease: {disease[0]}")
    st.info(f"Confidence: {confidence*100:.2f}%")