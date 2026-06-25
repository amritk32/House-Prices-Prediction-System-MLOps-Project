import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    return pd.read_csv(path)


def split_train_test(
    df: pd.DataFrame,
    target_col: str = "SalePrice",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split the raw DataFrame into train and test sets.

    The returned X_train and X_test are raw feature DataFrames.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test
