import json
import os

import pandas as pd
from flask import Flask, request, render_template

from heartattack_predictor import HeartAttackPredictor

#Flask constructor
app = Flask(__name__)
app.config["DEBUG"] = True

# A decorator used to tell the application which URL is associated function
@app.route('/predict_heartattack', methods=["GET", "POST"])
def predict_heartattack():
    if request.method == "POST":
        prediction_input = [
            {
                "exng": int(request.form.get("exng")),
                "caa": int(request.form.get("caa")),
                "cp": int(request.form.get("cp")),
                "fbs": int(request.form.get("fbs")),
                "restecg": int(request.form.get("restecg")),
                "slp": int(request.form.get("slp")),
                "thall": int(request.form.get("thall")),
                "trtbps": float(request.form.get("trtbps")),
                "chol": float(request.form.get("chol")),
                "thalachh": float(request.form.get("thalachh")),
                "oldpeak": float(request.form.get("oldpeak")),
                "age": int(request.form.get("age")),
                "sex": int(request.form.get("sex"))
            }
        ]
        print(prediction_input)
        hap = HeartAttackPredictor()
        df = pd.read_json(json.dumps(prediction_input), orient='records')
        status = hap.predict_single_record(df)
        #return the prediction outcome as a json message. 200 is HTTP status code 200, indicating successful completion
        return jsonify({'result': str(status[0])}), 200

    return render_template(
        "user_form.html") #this method is called of HTTP method is GET

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)
