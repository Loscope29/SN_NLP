import whisper
import os
from langsmith import traceable
from dotenv import load_dotenv


load_dotenv()  
## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"


# Chargement du modèle Whisper
model = whisper.load_model("base")

@traceable("AudioToText-Whisper")
def audio_to_text(audio_path):
    # Transcription de l'audio
    result = model.transcribe(audio_path)
    return result["text"]
