import streamlit as st
import os
import rag

st.title("RAG - Questions & Réponses sur PDF")
st.write("Téléchargez un PDF et posez une question.")

pdf_file = st.file_uploader("Choisir un fichier PDF", type=["pdf"])
question = st.text_input("Votre question:")

if pdf_file is not None and question:
    # Sauvegarder temporairement
    temp_path = "temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(pdf_file.getbuffer())
    
    if st.button("Obtenir la réponse"):
        with st.spinner("Analyse en cours..."):
            result = rag.main(temp_path, question)
        st.success("Réponse générée!")
        st.write("**Question:**", question)
        st.write("**Réponse:**", result["answer"])
    
    # Nettoyer le fichier temporaire
    if os.path.exists(temp_path):
        os.remove(temp_path)