import streamlit as st
import numpy as np
from scipy.io.wavfile import write
import io
import speech_audio_hf

st.title("Text to Speech avec VITS")
st.write("Entrez du texte pour générer de l'audio en français.")

text = st.text_area("Texte à convertir:", height=100)

if text and st.button("Générer l'audio"):
    with st.spinner("Génération en cours..."):
        audio = speech_audio_hf.text_to_speech(text)
    
    # Convertir en bytes pour Streamlit
    audio_bytes = io.BytesIO()
    sample_rate = speech_audio_hf.model.config.sampling_rate
    write(audio_bytes, sample_rate, audio.astype(np.float32))
    audio_bytes.seek(0)
    
    st.success("Audio généré!")
    st.audio(audio_bytes, format="audio/wav")
    
    # Télécharger l'audio
    st.download_button("Télécharger l'audio", audio_bytes.getvalue(), file_name="generated_audio.wav", mime="audio/wav")