# SN NLP Repository

Ce dépôt contient plusieurs projets Python autour du traitement du langage naturel (NLP), de la génération de texte assistée par récupération (RAG), de la conversion audio/texte et de la prédiction de séries temporelles financières.

## Aperçu des projets

- **RAG (Retrieval-Augmented Generation)**
  - `rag.py` : script permettant de charger un PDF, de créer une base de vecteurs et de répondre à des questions à partir du contenu du document.
  - `Rag_pdf.ipynb` : notebook de démonstration supplémentaire.
- **Speech-to-Text**
  - `audio_speech.ipynb` : transcription audio via Whisper.
- **Text-to-Speech**
  - `speech_audio.ipynb` : génération audio à partir de texte avec Edge TTS.
  - `speech_audio_hf.py` : synthèse vocale via un modèle Hugging Face.
- **Prédictions et analyse**
  - `Prediction_diabete.ipynb` : prédiction de diabète.
  - `Prediction_house_ANN.ipynb` : prédiction de prix immobiliers.
- **Projet Bourse**
  - `Bourse/lstm_prediction.ipynb` : modèle LSTM pour prédire les cours de clôture boursiers.
    - Le notebook utilise `yfinance` pour charger les données.
    - Le choix de l’action est défini par la variable `company_name`.
    - Le ticker est dérivé automatiquement de `company_name`.  - `Bourse/prediction_lstm.ipynb` : modèle LSTM pour prédire la direction des rendements boursiers.
    - Prédit si le prix va augmenter ou diminuer (directional accuracy).
    - Génère des fichiers `.npy` avec les prédictions.
    - Calcule des métriques de performance : directional accuracy (~50.89%), Sharpe ratio (~0.38).
  - `Bourse/chatbot.py` : chatbot IA pour analyser les prédictions boursières.
    - Charge les prédictions depuis les fichiers `.npy` générés.
    - Utilise l'API Gemini de Google pour générer des analyses investissements.
    - Intégré avec LangChain pour traçage et suivi des coûts via LangSmith.
## Prérequis

- Python 3.8 ou supérieur
- Git
- Environnement virtuel (recommandé)
- Clés API pour les services utilisés si nécessaire

## Installation

1. **Cloner le dépôt** :
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Activer l’environnement virtuel** :
   Le dépôt inclut un environnement dans `SN_NLP/`. Activez-le :
   ```bash
   # Sur Windows
   SN_NLP\Scripts\activate
   ```

3. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```
   Si `requirements.txt` n’existe pas, installez manuellement :
   ```bash
   pip install pandas numpy matplotlib yfinance tensorflow scikit-learn statsmodels langchain transformers torch scipy playsound edge-tts chromadb sentence-transformers python-dotenv
   ```

4. **Configurer les variables d’environnement** :
   Créez un fichier `.env` à la racine si nécessaire :
   ```ini
   # LangChain et LangSmith (optionnel mais recommandé pour traçage)
   LANGCHAIN_API_KEY=your_langsmith_api_key
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=your_project_name
   
   # Google Generative AI (requis pour chatbot.py)
   GEMINI_API_KEY=your_gemini_api_key
   ```
   Ajoutez `.env` à `.gitignore` pour éviter de versionner des secrets.

## Utilisation

### Choix d’une action dans `Bourse/lstm_prediction.ipynb`

Dans le notebook, changez simplement la variable :
```python
company_name = "General Electric"
```
La valeur de `ticker` sera mise à jour automatiquement grâce à :
```python
ticker = TICKERS.get(company_name, "AMZN").replace(" ", "_")
```

### Exécuter le notebook Bourse

1. Ouvrez le notebook :
   ```bash
   jupyter notebook Bourse/lstm_prediction.ipynb
   ```
2. Exécutez les cellules dans l’ordre.
3. Vérifiez la métrique `Price RMSE` et le graphique `Actual vs Predicted Stock Price`.
### Utiliser `prediction_lstm.ipynb` pour les prédictions de rendements

1. Ouvrez le notebook :
   ```bash
   jupyter notebook Bourse/prediction_lstm.ipynb
   ```
2. Ce notebook génère des fichiers `.npy` contenant les prédictions.
3. Les métriques calculées :
   - **Directional Accuracy** (~50.89%) : pourcentage de prédictions correctes sur la direction (hausse/baisse).
   - **Sharpe Ratio** (~0.38) : mesure de performance risque-ajustée (supérieur à 0 est bon).
4. Les fichiers de prédictions sont sauvegardés dans `Bourse/` (ex: `AMD_predictions.npy`).

### Utiliser le chatbot IA `Bourse/chatbot.py`

1. D'abord, générez les prédictions via `prediction_lstm.ipynb`.
2. Assurez-vous que votre clé API Gemini est configurée dans `.env` :
   ```ini
   GEMINI_API_KEY=your_gemini_api_key
   ```
3. Lancez le chatbot :
   ```bash
   python Bourse/chatbot.py
   ```
4. Fournissez le nom de l'action (ex: "Amazon", "AMD", "Google") ainsi que lo chemin vers le fichier de prédictions.
5. Le chatbot génère une analyse d'investissement basée sur les prédictions LSTM.
6. Les coûts API sont traçés via LangSmith (si `LANGCHAIN_API_KEY` est configurée).
### Utiliser le script RAG

1. Placez votre PDF dans `archive/`.
2. Lancez :
   ```bash
   python rag.py
   ```
3. Posez une question lorsque le script le demande.

### Utiliser les notebooks audio

#### `audio_speech.ipynb`
1. Lancez Jupyter :
   ```bash
   jupyter notebook audio_speech.ipynb
   ```
2. Exécutez les cellules pour transcrire un fichier audio.

#### `speech_audio.ipynb`
1. Ouvrez le notebook :
   ```bash
   jupyter notebook speech_audio.ipynb
   ```
2. Exécutez les cellules pour générer de l’audio à partir de texte.

#### `speech_audio_hf.py`
1. Lancez le script :
   ```bash
   python speech_audio_hf.py
   ```
2. Saisissez le texte à convertir en parole.

## Structure du projet

```
├── Bourse/                     # Projet de prédiction boursière
│   ├── lstm_prediction.ipynb    # Prédiction des prix de clôture
│   ├── prediction_lstm.ipynb    # Prédiction de rendements et directional accuracy
│   └── chatbot.py               # Chatbot IA pour analyser les prédictions
├── archive/                    # Documents PDF et fichiers de données
├── chroma_db/                  # Base de données de vecteurs
├── SN_NLP/                     # Environnement virtuel
├── audio_speech.ipynb          # Notebook speech-to-text
├── speech_audio.ipynb          # Notebook text-to-speech
├── rag.py                      # Script RAG
├── speech_audio_hf.py          # Script TTS Hugging Face
├── Rag_pdf.ipynb               # Notebook RAG supplémentaire
├── Prediction_diabete.ipynb    # Prédiction de diabète
├── Prediction_house_ANN.ipynb  # Prédiction immobilière
├── requirements.txt            # Dépendances Python
├── .env                        # Variables d’environnement (non suivi)
├── .gitignore                  # Fichiers ignorés par Git
└── README.md                   # Documentation du projet
```

## Suggestions d’amélioration

- Ajouter un `requirements.txt` propre avec toutes les dépendances du projet.
- Centraliser l’activation de l’environnement et la configuration `.env`.
- Harmoniser le pipeline `Bourse/lstm_prediction.ipynb` pour inversion complète des transformations.
- Ajouter un script ou un notebook pour tester automatiquement plusieurs tickers.

## Licence

Ce projet est à but éducatif. Vérifiez les licences des bibliothèques tierces utilisées.

## Aide

- Si vous rencontrez des erreurs d’installation, vérifiez l’environnement virtuel et les versions de Python.
- Si `yfinance` n’ouvre pas de données, assurez-vous que la connexion internet est active.
- Pour les erreurs de modèle TensorFlow, vérifiez que `tensorflow` est compatible avec votre version Python.

Pour toute question, consultez le code ou ajoutez une issue dans le dépôt.