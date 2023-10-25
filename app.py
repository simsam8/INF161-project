import numpy as np
import pandas as pd
import pickle


from flask import Flask, request, render_template
from waitress import serve

app = Flask(__name__)

# -- load model --
with open("models/model.pickle", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return render_template("./index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Render results
    """

    features = dict(request.form)

    form_columns = [
        "Dato",
        "Tid",
        "Globalstraling",
        "Solskinstid",
        "Lufttemperatur",
        "Vindretning",
        "Vindstyrke",
        "Lufttrykk",
        "Vindkast",
    ]

    numeric_features = form_columns[2:]

    def to_numeric(key, value, numeric_features=numeric_features):
        if key not in numeric_features:
            return value
        try:
            return float(value)
        except ValueError:
            return np.nan

    features = {key: to_numeric(key, value) for key, value in features.items()}
    print(features)
    forms_df = pd.DataFrame(features, columns=form_columns, index=[0])

    forms_df["hour"] = pd.to_datetime(forms_df["Tid"]).dt.hour
    forms_df["day"] = pd.to_datetime(forms_df["Dato"]).dt.weekday
    forms_df["month"] = pd.to_datetime(forms_df["Dato"]).dt.month

    features_df = forms_df.drop(["Dato", "Tid"], axis=1)
    print(features_df)

    prediction = model.predict(features_df)
    prediction = np.round(prediction[0])
    prediction = np.clip(prediction, 0, np.inf)
    print(prediction)

    return render_template(
        "./index.html", pred_text=f"Predikert sykkelvolum : {prediction}"
    )


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8080)
