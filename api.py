from fastapi import FastAPI
import pandas as pd
from pdantic import HouseInput
import joblib
import numpy as np
from feature_engineering import FeatureEngineer
app = FastAPI()

model = joblib.load("house_price_model.pkl")

@app.get('/')
def root():
    return {"message" : "House Price Prediction API Running..."}


@app.post("/predict")
def predict(data : HouseInput):
    feature_engineer = joblib.load('feature_engineer.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    model = joblib.load('house_price_model.pkl')
    template = joblib.load("default_template.pkl")

    input_df = template.copy()
    input_df["OverallQual"] = data.OverallQual
    input_df["GrLivArea"] = data.GrLivArea
    input_df["GarageCars"] = data.GarageCars
    input_df["TotalBsmtSF"] = data.TotalBsmtSF
    input_df["YearBuilt"] = data.YearBuilt

    engineered_df = feature_engineer.transform(input_df)

    transformed_df = preprocessor.transform(engineered_df)

    prediction_logs = model.predict(transformed_df)

    prediction = np.expm1(prediction_logs)

    return {"predicted_price " : float(prediction[0])}