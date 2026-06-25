import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_log_error


def evaluate_model(model, X_val, y_val, name: str):
    predictions_log = model.predict(X_val)
    predictions = np.expm1(predictions_log)
    y_val_real = np.expm1(y_val)

    rmsle = root_mean_squared_log_error(y_val_real, predictions)
    mse = mean_squared_error(y_val_real, predictions)
    r2 = r2_score(y_val_real, predictions)

    print(f"Model: {name}")
    print(f"RMSLE: {rmsle:.5f}")
    print(f"MSE: {mse:.5f}")
    print(f"R2 Score: {r2:.5f}")
    print("-----------------------")

    return {
        "model": name,
        "rmsle": rmsle,
        "mse": mse,
        "r2": r2,
    }
