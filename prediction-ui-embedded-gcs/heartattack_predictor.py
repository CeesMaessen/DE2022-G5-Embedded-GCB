import json
import os
import pandas as pd
import joblib
import sklearn
from sklearn.preprocessing import RobustScaler

class HeartAttackPredictor:
    def __init__(self):
        self.model = None

    def predict_single_record(self, prediction_input):
        print(prediction_input)
        cat_cols = ['sex','exng','caa','cp','fbs','restecg','slp','thall']
        con_cols = ["age","trtbps","chol","thalachh","oldpeak"]

        model_name = os.environ.get('MODEL_NAME', 'Specified environment variable is not set.')
        if self.model is None:
            self.model = joblib.load(model_name)

        df = pd.read_json((prediction_input).to_json(), orient='records')

        df = pd.get_dummies(df, columns = cat_cols, drop_first = True)

        # instantiate the scaler
        scaler = RobustScaler()

        # scaling the continuous features
        df[con_cols] = scaler.fit_transform(df[con_cols])

        y_pred = self.model.predict(df)
        return y_pred
