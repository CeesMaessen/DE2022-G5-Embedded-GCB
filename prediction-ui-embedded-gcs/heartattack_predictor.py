import json
import os
import pandas as pd
import joblib
import sklearn

class HeartAttackPredictor:
    def __init__(self):
        self.model = None

    def predict_single_record(self, prediction_input):
        print(prediction_input)
        model_name = os.environ.get('MODEL_NAME', 'Specified environment variable is not set.')
        if self.model is None:
            self.model = joblib.load(model_name)
        df = pd.read_json(to_json(prediction_input), orient='records')
        y_pred = self.model.predict(df)
        return y_pred
