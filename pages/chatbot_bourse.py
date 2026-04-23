

import streamlit as st
from Bourse import chatbot
from Bourse import prediction_utils


st.title("Chatbot Financier pour la Bourse")
st.write("Choisissez une entreprise et un modèle de prédiction, puis posez votre question.")

company = st.selectbox("Choisir l'entreprise:", ["AMD", "GE", "AMZN", "INTC"])
model_type = st.selectbox("Choisir le modèle:", ["Hugging Face (Chronos)", "Local LSTM"])
question = st.text_input("Votre question:")

if question:
    if st.button("Obtenir la réponse"):
        with st.spinner("Génération des prédictions et de la réponse..."):
            try:
                if model_type == "Hugging Face (Chronos)":
                    predictions = prediction_utils.predict_with_hf(company)
                else:
                    predictions = prediction_utils.predict_with_lstm(company)
                
                response = chatbot.get_chatbot_response(company, predictions.tolist(), question)
                st.success("Réponse générée!")
                st.write("**Entreprise:**", company)
                st.write("**Modèle:**", model_type)
                st.write("**Prédictions (prix futurs):**", f"{len(predictions)} jours, dernier prix prédit: {predictions[-1]:.2f} $")
                st.write("**Question:**", question)
                st.write("**Réponse:**", response)
            except Exception as e:
                st.error(f"Erreur: {str(e)}")