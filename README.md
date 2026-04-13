# NLP Projects Repository

This repository contains various NLP (Natural Language Processing) projects implemented in Python, including Retrieval-Augmented Generation (RAG), speech-to-text, and text-to-speech functionalities.

## Projects Overview

- **RAG (Retrieval-Augmented Generation)**: `rag.py` - A system that combines document retrieval with language model generation for answering questions based on PDF documents.
- **Speech-to-Text**: `audio_speech.ipynb` - Notebook for transcribing audio files using Whisper.
- **Text-to-Speech**: `speech_audio.ipynb` and `speech_audio_hf.py` - Notebooks and scripts for generating audio from text using Edge TTS and Hugging Face models.
- **Other Notebooks**: Various machine learning models for predictions (diabetes, house prices) and data analysis.

## Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment (recommended)
- API keys for LangSmith (for tracking) and other services if needed

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set up the virtual environment**:
   The repository includes a virtual environment in `SN_NLP/`. Activate it:
   ```bash
   # On Windows
   SN_NLP\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` doesn't exist, install manually:
   ```bash
   pip install langchain langchain-community langchain-huggingface langchain-ollama transformers torch scipy playsound edge-tts chromadb sentence-transformers dotenv langsmith
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory with your API keys:
   ```
   LANGCHAIN_API_KEY=your_langsmith_api_key
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=your_project_name
   ```
   **Important**: Add `.env` to `.gitignore` to avoid committing secrets.

## Usage

### RAG System (`rag.py`)

This script loads a PDF, creates a vector store, and answers questions based on the document content.

1. Place your PDF in the `archive/` directory (e.g., `DS interview quESTIONS.pdf`).
2. Run the script:
   ```bash
   python rag.py
   ```
3. The script will ask for a question and provide an answer based on the PDF content.

### Speech-to-Text (`audio_speech.ipynb`)

1. Open the notebook in Jupyter:
   ```bash
   jupyter notebook audio_speech.ipynb
   ```
2. Run the cells to load Whisper model and transcribe audio.
3. Place your audio file in the appropriate path and update the file path in the notebook.

### Text-to-Speech (`speech_audio.ipynb`)

1. Open the notebook:
   ```bash
   jupyter notebook speech_audio.ipynb
   ```
2. Run cells to generate audio from text using Edge TTS.
3. The audio will be saved as `reponse_assistant.mp3` and played automatically.

### Text-to-Speech with Hugging Face (`speech_audio_hf.py`)

1. Run the script:
   ```bash
   python speech_audio_hf.py
   ```
2. Enter the text to convert to speech.
3. The audio will be generated using VITS model, saved as `output_audio.wav`, and played.

### Other Notebooks

- `Prediction_diabete.ipynb`: ANN model for diabetes prediction.
- `Prediction_house_ANN.ipynb`: ANN model for house price prediction.
- `Rag_pdf.ipynb`: Additional RAG implementation.

Open these in Jupyter and run cells as needed.

## Project Structure

```
├── archive/                    # PDF documents and data files
├── chroma_db/                  # Vector database storage
├── SN_NLP/                     # Virtual environment
├── audio_speech.ipynb          # Speech-to-text notebook
├── speech_audio.ipynb          # Text-to-speech notebook
├── rag.py                      # RAG script
├── speech_audio_hf.py          # Hugging Face TTS script
├── Rag_pdf.ipynb               # Additional RAG notebook
├── Prediction_diabete.ipynb    # Diabetes prediction
├── Prediction_house_ANN.ipynb  # House price prediction
├── .env                        # Environment variables (not committed)
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Test thoroughly.
5. Submit a pull request.

## License

This project is for educational purposes. Please check individual licenses for third-party libraries used.

## Troubleshooting

- **Git Push Issues**: If you encounter push protection errors due to secrets in `.env`, follow the steps in the terminal error message or remove secrets from Git history.
- **Model Loading**: Ensure you have sufficient RAM/VRAM for large models like Whisper or VITS.
- **API Keys**: Verify your `.env` file is correctly configured and `.env` is in `.gitignore`.

For any issues, check the code comments or open an issue in the repository.