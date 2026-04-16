# SN NLP Repository

Ce dépôt regroupe plusieurs projets Python autour du traitement du langage naturel, de la génération multimodale, de l’analyse de données et de la prédiction financière.

## Aperçu du projet principal

Cette application fonctionne comme un frontend Streamlit centralisé, avec plusieurs pages :

- **Page d’accueil** : `app_principale.py`
- **Audio to Text** : `pages/audio_to_text.py`
- **RAG (PDF Q&A)** : `pages/rag_qa.py`
- **Text to Speech** : `pages/text_to_speech.py`
- **Chatbot Bourse** : `pages/chatbot_bourse.py`

Chaque page appelle les fonctions logiques des scripts existants sans casser les dépendances.

## Fonctionnalités

- **Transcription audio** : charger un fichier audio et obtenir une transcription via Whisper.
- **RAG sur PDF** : poser des questions à un document PDF à l’aide d’un pipeline de récupération de contexte.
- **Génération vocale** : convertir du texte en audio via un modèle Hugging Face.
- **Chatbot Bourse** : sélectionner une entreprise et un modèle de prédiction, puis générer une réponse basée sur les données de prévision.

## Autres scripts et notebooks

- `rag.py` / `Rag_pdf.ipynb` : RAG pour le question-answering sur PDF.
- `audio_speech.ipynb` : transcription audio avec Whisper.
- `speech_audio.ipynb` : génération audio à partir de texte avec Edge TTS.
- `speech_audio_hf.py` : génération audio via un modèle HF.
- `Prediction_diabete.ipynb` : prédiction de diabète.
- `Prediction_house_ANN.ipynb` : prédiction de prix immobiliers.
- `Bourse/lstm_prediction.ipynb` : modèle LSTM pour prédire les prix de clôture.
- `Bourse/prediction_lstm.ipynb` : prédiction de la direction des rendements boursiers.
- `Bourse/chatbot.py` : chatbot financier basé sur des prédictions.

## Prérequis

- Python 3.8 ou supérieur
- Git
- Environnement virtuel (recommandé)
- Connexion internet pour `yfinance`, les APIs et le téléchargement de modèles
- Clés API pour le chatbot :
  - `GEMINI_API_KEY` pour Google Generative AI
  - `LANGCHAIN_API_KEY` pour LangSmith (facultatif)

## Installation

1. **Cloner le dépôt** :
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Activer l’environnement virtuel** :
    ```bash
    # Sur Windows
    SN_NLP\Scripts\activate
    ```

3. **Installer les dépendances** :
    ```bash
    pip install -r requirements.txt
    ```

4. **Configurer les variables d’environnement** :
    Créez un fichier `.env` à partir d’un exemple ou directement dans le dossier principal ou `Bourse/` :
    ```bash
    copy Bourse\.env.example .env
    ```
    Ajoutez les clés :
    ```ini
    GEMINI_API_KEY=your_gemini_api_key
    LANGCHAIN_API_KEY=your_langsmith_api_key
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_PROJECT=your_project_name
    ```

    **Ne versionnez pas `.env`**.

## Utilisation

### Lancer l’application Streamlit

1. Activez l’environnement virtuel.
2. Lancez :
    ```bash
    streamlit run app_principale.py
    ```
3. Utilisez la barre latérale pour changer de page.

### Chatbot Bourse

- Choisissez une entreprise parmi : `AMD`, `GE`, `AMZN`, `INTC`.
- Choisissez le modèle de prédiction :
  - `Hugging Face (Chronos)` pour les prédictions provenant de Chronos.
  - `Local LSTM` pour les modèles Keras sauvegardés dans `Bourse/models/`.
- Posez votre question, puis générez la réponse.

### Audio to Text

- Téléversez un fichier audio.
- Transcrivez le texte.
- Téléchargez la transcription.

### Text to Speech

- Entrez du texte.
- Générez l’audio et écoutez-le dans le navigateur.
- Téléchargez le fichier `.wav`.

### RAG PDF

- Téléversez un PDF.
- Posez une question sur le document.
- Le système utilise un pipeline de récupération de contexte pour répondre.

## Structure du projet

```
SN NLP/
├── app_principale.py
├── pages/
│   ├── audio_to_text.py
│   ├── rag_qa.py
│   ├── text_to_speech.py
│   └── chatbot_bourse.py
├── audio_text.py
├── rag.py
├── speech_audio_hf.py
├── Bourse/
│   ├── chatbot.py
│   ├── prediction_utils.py
│   ├── lstm_prediction.ipynb
│   ├── prediction_lstm.ipynb
│   └── models/
├── archive/
├── chroma_db/
├── requirements.txt
└── README.md
```

## Notes importantes

- Vérifiez que `Bourse/models/` contient bien les fichiers de modèles nécessaires.
- Assurez-vous que `GEMINI_API_KEY` est configurée avant d’utiliser le chatbot financier.
- Si vous utilisez `Chronos`, la qualité des prédictions longues peut être moins bonne au-delà de 64 jours.

## Support

- Pour une erreur d’installation : vérifiez le venv et les paquets installés.
- Pour `yfinance` : assurez-vous d’avoir internet.
- Pour le chatbot : vérifiez les clés API dans `.env`.

Pour toute question, consultez le code ou ajoutez une issue dans le dépôt.
