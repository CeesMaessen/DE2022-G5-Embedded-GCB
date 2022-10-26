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
        catvals = [0,1,2,3,4]

        model_name = os.environ.get('MODEL_NAME', 'Specified environment variable is not set.')
        if self.model is None:
            self.model = joblib.load(model_name)

        df = pd.read_json((prediction_input).to_json(), orient='records')

        df_dummified = pd.get_dummies(df[cat_cols].astype(pd.CategoricalDtype(categories=catvals)))
        df = pd.concat([df, df_dummified], axis = 1)
        cols_to_drop = cat_cols + ['sex_0','sex_2','sex_3','sex_4','exng_0','exng_2','exng_3','exng_4','caa_0','cp_0','cp_4','fbs_0','fbs_2', 'fbs_3','fbs_4','restecg_0','restecg_3','restecg_4','slp_0', 'slp_3','slp_4', 'thall_0','thall_4']
        df = df.drop(columns = cols_to_drop)

        # instantiate the scaler
        scaler = RobustScaler()

        # scaling the continuous features
        df[con_cols] = scaler.fit_transform(df[con_cols])

        y_pred = self.model.predict(df)
        return y_pred
