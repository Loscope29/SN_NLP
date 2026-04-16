import torch
import numpy as np
import yfinance as yf
from chronos import ChronosPipeline
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os

TICKERS = {
    "AMD": "AMD",
    "GE": "GE",
    "AMZN": "AMZN",
    "INTC": "INTC"
}

ROOT_DIR = os.path.dirname(__file__)


def load_recent_data(ticker, period="2y"):
    data = yf.download(ticker, period=period)
    data.reset_index(inplace=True)
    if isinstance(data.columns, pd.MultiIndex):
        close = data[("Close", ticker)]
    else:
        close = data["Close"]
    return close.values.flatten()

def predict_with_hf(company, months=6):
    ticker = TICKERS[company]
    price = np.asarray(load_recent_data(ticker), dtype=np.float32)
    if price.ndim == 1:
        price = price.reshape(1, -1)
    context = torch.from_numpy(price)
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        dtype=torch.float32,
    )
    n_days = months * 21
    forecast = pipeline.predict(
        context,
        prediction_length=n_days,
        num_samples=20,
    )
    median_forecast = forecast[0].median(dim=0).values.numpy()
    return median_forecast

def predict_with_lstm(company, days=126):
    ticker = TICKERS[company]
    close_prices = load_recent_data(ticker, "10y")
    close_series = pd.Series(close_prices.flatten())

    # Préparation des données comme dans le notebook
    data_log = np.log(close_series)
    data_sqrt = np.sqrt(data_log)
    data_sqrt_diff = data_sqrt.diff().dropna()

    time_step = 60
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = data_sqrt_diff.values.reshape(-1, 1)
    scaler.fit(dataset)

    # Charger le modèle depuis le dossier Bourse/models relatif à ce fichier
    model_path_keras = os.path.join(ROOT_DIR, "models", f"{company}_best_model.keras")
    model_path_h5 = os.path.join(ROOT_DIR, "models", f"{company}_best_model.h5")
    if os.path.exists(model_path_keras):
        model_path = model_path_keras
    elif os.path.exists(model_path_h5):
        model_path = model_path_h5
    else:
        raise FileNotFoundError(f"Modèle pour {company} non trouvé dans {ROOT_DIR}/models.")
    model = load_model(model_path)

    # Préparer la dernière séquence
    last_sequence = dataset[-time_step:].reshape(1, time_step, 1)
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1)).reshape(last_sequence.shape)

    predictions_diff = []
    current_sequence = last_sequence_scaled.copy()

    for _ in range(days):
        pred_scaled = model.predict(current_sequence, verbose=0)
        predictions_diff.append(pred_scaled[0][0])

        # Ajouter la prédiction à la séquence pour la prochaine
        new_point = np.array([[pred_scaled[0][0]]]).reshape(1, 1, 1)
        current_sequence = np.concatenate([current_sequence[:, 1:, :], new_point], axis=1)

    # Inverser les différences
    predictions_diff = np.array(predictions_diff).reshape(-1, 1)
    predictions_diff_unscaled = scaler.inverse_transform(predictions_diff).flatten()

    # Accumuler les différences à partir du dernier sqrt réel
    last_sqrt = data_sqrt.iloc[-1]
    predicted_sqrt = [last_sqrt]
    for diff in predictions_diff_unscaled:
        last_sqrt = last_sqrt + diff
        predicted_sqrt.append(last_sqrt)

    # Inverser sqrt puis log pour retrouver les prix
    predicted_log = np.square(np.array(predicted_sqrt))
    predicted_prices = np.exp(predicted_log)

    return predicted_prices[1:]  # Exclure le dernier prix réel du début