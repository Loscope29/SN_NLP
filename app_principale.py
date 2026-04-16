import streamlit as st

st.set_page_config(page_title="SN NLP App", layout="wide")

st.title("Bienvenue dans l'App SN NLP")
st.write("Utilisez la barre latérale pour naviguer entre les différentes fonctionnalités.")

st.markdown("""
Cette application intègre plusieurs outils NLP :

- **Audio to Text** : Transcrivez des fichiers audio en texte avec Whisper.
- **RAG (PDF Q&A)** : Posez des questions sur des documents PDF.
- **Text to Speech** : Générez de l'audio à partir de texte en français.
- **Chatbot Bourse** : Analysez des prédictions financières avec un chatbot IA.

Sélectionnez une page dans la barre latérale pour commencer.
""")

st.sidebar.success("Sélectionnez une page ci-dessus.")