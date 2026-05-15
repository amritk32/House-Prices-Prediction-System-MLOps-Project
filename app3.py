from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

# =========================================
# LOAD MODEL
# =========================================

model = joblib.load("house_price_model.pkl")

model_features = joblib.load(
    "model_features.pkl"
)

# =========================================
# INPUT SCHEMA
# =========================================

class HouseInput(BaseModel):

    MSSubClass: float
    MSZoning: str
    LotFrontage: float
    LotArea: float
    Street: str
    Alley: str
    LotShape: str
    LandContour: str
    Utilities: str
    LotConfig: str
    LandSlope: str
    Neighborhood: str
    Condition1: str
    Condition2: str
    BldgType: str
    HouseStyle: str
    OverallQual: float
    OverallCond: float
    YearBuilt: float
    YearRemodAdd: float
    RoofStyle: str
    RoofMatl: str
    Exterior1st: str
    Exterior2nd: str
    MasVnrType: str
    MasVnrArea: float
    ExterQual: str
    ExterCond: str
    Foundation: str
    BsmtQual: str
    BsmtCond: str
    BsmtExposure: str
    BsmtFinType1: str
    BsmtFinSF1: float
    BsmtFinType2: str
    BsmtFinSF2: float
    BsmtUnfSF: float
    TotalBsmtSF: float
    Heating: str
    HeatingQC: str
    CentralAir: str
    Electrical: str
    FirstFlrSF: float
    SecondFlrSF: float
    LowQualFinSF: float
    GrLivArea: float
    BsmtFullBath: float
    BsmtHalfBath: float
    FullBath: float
    HalfBath: float
    BedroomAbvGr: float
    KitchenAbvGr: float
    KitchenQual: str
    TotRmsAbvGrd: float
    Functional: str
    Fireplaces: float
    FireplaceQu: str
    GarageType: str
    GarageFinish: str
    GarageCars: float
    GarageArea: float
    GarageQual: str
    GarageCond: str
    PavedDrive: str
    WoodDeckSF: float
    OpenPorchSF: float
    EnclosedPorch: float
    ThreeSsnPorch: float
    ScreenPorch: float
    PoolArea: float
    PoolQC: str
    Fence: str
    MiscFeature: str
    MiscVal: float
    MoSold: float
    YrSold: float
    SaleType: str
    SaleCondition: str

# =========================================
# FEATURE ENGINEERING
# =========================================

def prepare_features(data):

    df = pd.DataFrame([data])

    # SAME FEATURE ENGINEERING
    df['GarageAge'] = df['YrSold'] - df['YearBuilt']

    df['TotalArea'] = (
        df['GrLivArea']
        + df['GarageArea']
        + df['LotArea']
        + df['TotalBsmtSF']
    )

    df['Bath'] = (
        df['FullBath']
        + 0.5 * df['HalfBath']
        + df['BsmtFullBath']
        + 0.5 * df['BsmtHalfBath']
    )

    df['Age'] = df['YrSold'] - df['YearBuilt']

    df['QualArea'] = (
        df['OverallQual']
        + df['GrLivArea']
    )

    df['HasPool'] = (
        df['PoolArea'] > 0
    ).astype(int)

    # MATCH COLUMN ORDER
    df = df.reindex(
        columns=model_features,
        fill_value=0
    )

    return df

# =========================================
# PREDICTION ENDPOINT
# =========================================

@app.post("/predict")

def predict_price(data: HouseInput):

    input_df = prepare_features(
        data.dict()
    )

    prediction_log = model.predict(input_df)[0]

    prediction = np.expm1(prediction_log)

    return {
        "predicted_house_price": float(prediction)
    }