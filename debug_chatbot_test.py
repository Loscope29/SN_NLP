import traceback
from Bourse import prediction_utils

print('import ok')

try:
    preds = prediction_utils.predict_with_hf('AMD')
    print('HF len', len(preds))
    print('HF first', preds[:3])
except Exception as e:
    traceback.print_exc()

try:
    preds2 = prediction_utils.predict_with_lstm('AMD')
    print('LSTM len', len(preds2))
    print('LSTM first', preds2[:3])
except Exception as e:
    traceback.print_exc()
