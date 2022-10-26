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
        catvals = [0,1,2,3]

        model_name = os.environ.get('MODEL_NAME', 'Specified environment variable is not set.')
        if self.model is None:
            self.model = joblib.load(model_name)

        df = pd.read_json((prediction_input).to_json(), orient='records')

        df_dummified = pd.get_dummies(df[cat_cols].astype(pd.CategoricalDtype(categories=catvals)))
        df_dummified.insert(0, con_cols, df[con_cols])

        # instantiate the scaler
        scaler = RobustScaler()

        # scaling the continuous features
        df_dummified[con_cols] = scaler.fit_transform(df[con_cols])

        y_pred = self.model.predict(df_dummified)
        return y_pred
