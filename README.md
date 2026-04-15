# SN NLP Repository

Ce dépôt regroupe plusieurs projets Python autour du traitement du langage naturel, de la génération multimodale, de l’analyse de données et de la prédiction financière.

## Aperçu des projets

- **RAG (Retrieval-Augmented Generation)**
  - `rag.py` : script pour charger un PDF, créer une base de vecteurs et répondre à des questions via recherche contextuelle.
  - `Rag_pdf.ipynb` : notebook de démonstration complémentaire.
- **Speech-to-Text**
  - `audio_speech.ipynb` : transcription audio avec Whisper.
- **Text-to-Speech**
  - `speech_audio.ipynb` : génération audio à partir de texte avec Edge TTS.
  - `speech_audio_hf.py` : synthèse vocale via un modèle Hugging Face.
- **Prédiction et analyse**
  - `Prediction_diabete.ipynb` : prédiction de diabète.
  - `Prediction_house_ANN.ipynb` : prédiction de prix immobiliers.
- **Projet Bourse**
  - `Bourse/lstm_prediction.ipynb` : modèle LSTM pour prédire les cours de clôture boursiers.
    - Utilise `yfinance` pour charger les données.
    - Choix de l’action via la variable `company_name`.
    - Le ticker est dérivé automatiquement de `company_name`.
  - `Bourse/prediction_lstm.ipynb` : modèle LSTM pour prédire la direction des rendements boursiers.
    - Prédit la direction de la variation de prix (hausse ou baisse).
    - Calcule des métriques de performance : directional accuracy et Sharpe ratio.
    - Produit des fichiers de prédictions au format NumPy.
  - `Bourse/chatbot.py` : chatbot IA pour analyser et expliquer les prédictions.
    - Charge les prédictions sauvegardées dans `Bourse/models/predicted_prices.npy`.
    - Utilise l’API Gemini de Google via LangChain.
    - Suit les coûts API via LangSmith si les clés sont configurées.

## Prérequis

- Python 3.8 ou supérieur
- Git
- Environnement virtuel (recommandé)
- Connexion internet pour `yfinance` et les APIs cloud
- Clés API pour le chatbot :
 - `GEMINI_API_KEY` pour Google Generative AI
 - `LANGCHAIN_API_KEY` pour LangSmith (facultatif mais recommandé)

## Installation

1. **Cloner le dépôt** :
 ```bash
 git clone <repository-url>
 cd <repository-directory>
 ```

2. **Activer l’environnement virtuel** :
 Le dépôt contient un environnement Python dans `SN_NLP/`. Activez-le :
 ```bash
 # Sur Windows
 SN_NLP\Scripts\activate
 ```

3. **Installer les dépendances** :
 ```bash
 pip install -r requirements.txt
 ```
 Si le fichier est incomplet, installez manuellement :
 ```bash
 pip install pandas numpy matplotlib yfinance tensorflow scikit-learn statsmodels langchain transformers torch scipy playsound edge-tts chromadb sentence-transformers python-dotenv
 ```

4. **Configurer les variables d’environnement** :
 Copiez `Bourse/.env.example` vers `.env` et éditez-le :
 ```bash
 copy Bourse\.env.example .env
 ```
 Puis renseignez vos clés :
 ```ini
 GEMINI_API_KEY=your_gemini_api_key
 LANGCHAIN_API_KEY=your_langsmith_api_key
 LANGCHAIN_TRACING_V2=true
 LANGCHAIN_PROJECT=your_project_name
 ```
 **Ne versionnez jamais `.env`**. Il est ignoré par Git.

## Utilisation

### 1. Notebook `Bourse/lstm_prediction.ipynb`

Ce notebook entraîne un LSTM pour prédire le prix de clôture des actions.

1. Ouvrez :
 ```bash
 jupyter notebook Bourse/lstm_prediction.ipynb
 ```
2. Exécutez les cellules dans l’ordre.
3. Modifiez `company_name` pour tester d’autres sociétés.
4. Vérifiez les métriques de performance et les graphiques.

### 2. Notebook `Bourse/prediction_lstm.ipynb`

Ce notebook travaille sur la prédiction de la direction des rendements.

1. Ouvrez :
 ```bash
 jupyter notebook Bourse/prediction_lstm.ipynb
 ```
2. Exécutez les cellules.
3. Le notebook calcule :
 - **Directional Accuracy** : précision de la direction de la variation.
 - **Sharpe Ratio** : performance ajustée au risque.
4. Les prédictions peuvent être sauvegardées dans `Bourse/models/predicted_prices.npy`.

### 3. Script `Bourse/chatbot.py`

1. Vérifiez que les prédictions existent dans `Bourse/models/predicted_prices.npy`.
2. Assurez-vous que la clé `GEMINI_API_KEY` est présente dans `.env`.
3. Lancez :
 ```bash
 python Bourse/chatbot.py
 ```
4. Le chatbot émet une analyse basée sur les prédictions LSTM.
5. Si `LANGCHAIN_API_KEY` est défini, le suivi des coûts via LangSmith est activé.

### 4. Script `rag.py`

1. Placez votre PDF dans `archive/`.
2. Lancez :
 ```bash
 python rag.py
 ```
3. Posez une question lorsque le script le demande.

### 5. Notebooks audio

#### `audio_speech.ipynb`

1. Ouvrez :
 ```bash
 jupyter notebook audio_speech.ipynb
 ```
2. Exécutez les cellules pour transcrire un fichier audio.

#### `speech_audio.ipynb`

1. Ouvrez :
 ```bash
 jupyter notebook speech_audio.ipynb
 ```
2. Exécutez les cellules pour générer de l’audio à partir de texte.

#### `speech_audio_hf.py`

1. Lancez :
 ```bash
 python speech_audio_hf.py
 ```
2. Saisissez le texte à convertir en parole.

## Structure du projet

```
├── Bourse/ # Projet de prédiction boursière
│ ├── lstm_prediction.ipynb # Prédiction des prix de clôture
│ ├── prediction_lstm.ipynb # Prédiction de rendements et directional accuracy
│ ├── chatbot.py # Chatbot IA pour analyser les prédictions
│ ├── models/ # Fichiers de prédictions et modèles
│ └── .env.example # Exemple de configuration des clés API
├── archive/ # Documents PDF et fichiers de données
├── chroma_db/ # Base de données de vecteurs
├── SN_NLP/ # Environnement virtuel
├── audio_speech.ipynb # Notebook speech-to-text
├── speech_audio.ipynb # Notebook text-to-speech
├── rag.py # Script RAG
├── speech_audio_hf.py # Script TTS Hugging Face
├── Rag_pdf.ipynb # Notebook RAG supplémentaire
├── Prediction_diabete.ipynb # Prédiction de diabète
├── Prediction_house_ANN.ipynb # Prédiction immobilière
├── requirements.txt # Dépendances Python
├── .gitignore # Fichiers ignorés par Git
└── README.md # Documentation du projet
```

## Recommandations

- Ne mettez pas le fichier `.env` dans le dépôt.
- Utilisez `Bourse/.env.example` comme base de configuration.
- Exécutez les notebooks de génération de données avant d’utiliser le chatbot.
- Assurez-vous que la clé `GEMINI_API_KEY` est valide avant d’exécuter `Bourse/chatbot.py`.

## Licence

Ce projet est à but éducatif. Vérifiez les licences des bibliothèques tierces utilisées.

## Aide

- Si vous rencontrez des erreurs d’installation, vérifiez l’environnement virtuel et la version de Python.
- Si `yfinance` ne charge pas de données, assurez-vous que la connexion internet est active.
- Pour les erreurs TensorFlow, utilisez une version compatible avec votre installation Python.

Pour toute question, consultez le code ou ouvrez une issue dans le dépôt.
