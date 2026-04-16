from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import numpy as np
from langsmith import traceable

load_dotenv()

# Set environment variables for API keys
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ['LANGSMITH_PROJECT'] = 'Bourse Chatbot'

@traceable
def get_chatbot_response(company, predictions, user_question):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    if len(predictions) == 0:
        return "Erreur : Aucune donnée de prédiction disponible."

    last_price = predictions[-1]
    first_price = predictions[0]
    if first_price == 0:
        variation = 0
    else:
        variation = ((last_price - first_price) / first_price) * 100
    trend = "hausse 📈" if variation > 0 else "baisse 📉"

    system_prompt = f"""Tu es un assistant financier expert.Tu aides les utilisateurs à comprendre les tendances du marché boursier en te basant sur des données de prédiction. 


Voici les données de prédiction pour {company} :
- Prix prédit dans {len(predictions)//21} mois : {last_price:.2f} $
- Variation estimée : {variation:.2f}% ({trend})
- Tendance générale : {trend}

Réponds aux questions de l'utilisateur en te basant sur ces données.
Rappelle toujours que ce sont des prédictions, pas des certitudes."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]

    response = llm.invoke(messages)
    usage = response.usage_metadata
    print(f"Tokens utilisés : {usage.get('total_tokens', 'N/A')}, Coût estimé : ${usage.get('total_cost', 'N/A')}")
    return response.content

# Chargement des prédictions sauvegardées
script_dir = os.path.dirname(__file__)
predictions_path = os.path.join(script_dir, "models", "predicted_prices.npy")

try:
    predictions = np.load(predictions_path)
    print(f"Prédictions chargées depuis {predictions_path}")
except FileNotFoundError:
    print(f"Erreur : Fichier de prédictions non trouvé à {predictions_path}")
    print("Veuillez entraîner le modèle et sauvegarder les prédictions.")
    predictions = np.array([])  # Placeholder vide

if __name__ == "__main__":
    company = "Amazon"
    user_question = "Quelle est la tendance générale pour Amazon dans les prochains mois ? \
    Donne des actions a faire pour un investisseur qui veut investir dans Amazon ?"
    response = get_chatbot_response(company, predictions, user_question)
    print("Chatbot response:", response)