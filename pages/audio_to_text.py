import streamlit as st
import os
import audio_text

st.title("Audio to Text avec Whisper")
st.write("Téléchargez un fichier audio pour le transcrire en texte.")

audio_file = st.file_uploader("Choisir un fichier audio", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    # Sauvegarder temporairement
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(audio_file.getbuffer())
    
    if st.button("Transcrire"):
        with st.spinner("Transcription en cours..."):
            transcription = audio_text.audio_to_text(temp_path)
        st.success("Transcription terminée!")
        st.text_area("Transcription:", transcription, height=200)
        
        # Télécharger la transcription
        st.download_button("Télécharger la transcription", transcription, file_name="transcription.txt", mime="text/plain")
        
        # Lire la transcription (optionnel, mais c'est du texte)
        if st.button("Lire la transcription (TTS)"):
            # Ici, on pourrait intégrer text_to_speech pour lire le texte, mais pour l'instant, juste afficher
            st.write("Fonctionnalité de lecture à implémenter.")
    
    # Nettoyer le fichier temporaire
    if os.path.exists(temp_path):
        os.remove(temp_path)