import joblib
import numpy as np
import pandas as pd
from data_split import load_data, split_train_test
from feature_engineering import FeatureEngineer, clean_dataset
from model_runner import (
    get_lgbm_model,
    get_ridge_model,
    get_voting_ensemble,
    get_xgboost_model,
    train_model,
)
from model_evaluation import evaluate_model
from preprocessor import preprocess_data
pd.set_option('display.max_rows', None) 

DATA_PATH = "train.csv"
TARGET_COLUMN = "SalePrice"


def main():
    df = load_data(DATA_PATH)
    df = clean_dataset(df)

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = split_train_test(
        df, target_col=TARGET_COLUMN
    )
    default_row = X_train_raw.iloc[[0]].copy()
    joblib.dump(default_row, "default_template.pkl")
    feature_engineer = FeatureEngineer()
    feature_engineer.fit(X_train_raw)

    X_train = feature_engineer.transform(X_train_raw)
    X_test = feature_engineer.transform(X_test_raw)

    # print(X_train.dtypes)

    y_train = np.log1p(y_train_raw)
    y_test = np.log1p(y_test_raw)

    X_train_processed, X_test_processed, preprocessor = preprocess_data(
        X_train, X_test
    )

    xgb_model = train_model(get_xgboost_model(), X_train_processed, y_train)
    ridge_model = train_model(get_ridge_model(), X_train_processed, y_train)
    lgbm_model = train_model(get_lgbm_model(), X_train_processed, y_train)

    voting_ensemble = get_voting_ensemble(
        estimators=[
            ("xgb", get_xgboost_model()),
            ("ridge", get_ridge_model()),
            ("lgbm", get_lgbm_model()),
        ],
        weights=[1, 3, 1],
    )
    voting_ensemble = train_model(voting_ensemble, X_train_processed, y_train)

    evaluate_model(xgb_model, X_test_processed, y_test, "XGBoost")
    evaluate_model(ridge_model, X_test_processed, y_test, "Ridge")
    evaluate_model(lgbm_model, X_test_processed, y_test, "LightGBM")
    final_metrics = evaluate_model(voting_ensemble, X_test_processed, y_test, "Voting Ensemble")

    joblib.dump(voting_ensemble, "house_price_model.pkl")
    joblib.dump(preprocessor, "preprocessor.pkl")
    joblib.dump(feature_engineer, "feature_engineer.pkl")
    joblib.dump(X_train.columns.tolist(), "model_features.pkl")

    joblib.dump(final_metrics, "metrics.pkl")

    print("✅ House Price Model Saved Successfully!")


if __name__ == "__main__":
    main()
