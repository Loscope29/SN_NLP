from transformers import VitsModel, AutoTokenizer
from langsmith import traceable
from dotenv import load_dotenv
load_dotenv()  # Charge les variables d'environnement depuis le fichier .env
import os
import torch
import numpy as np
from scipy.io.wavfile import write


## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")


# Chargement du modèle et du processeur
model = VitsModel.from_pretrained("facebook/mms-tts-fra")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-fra")


@traceable("TextToSpeech-VITS")
def text_to_speech(text):
    # Génération de l'audio
    inputs = tokenizer(text, return_tensors="pt", voice_preset="v2/fr_speaker_1")
    audio = model(**inputs).waveform.squeeze().detach().cpu().numpy()
    return audio