import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw DataFrame before splitting into train and test."""
    return df[df["SaleCondition"] != "Partial"].copy()


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lotfrontage_neighborhood_median_ = None
        self.masvnrarea_median_ = None
        self.garageyrbult_median_ = None
        self.bsmt_medians_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.lotfrontage_neighborhood_median_ = (
            X.groupby("Neighborhood")["LotFrontage"].median()
        )
        self.masvnrarea_median_ = X["MasVnrArea"].median()
        self.garageyrbult_median_ = X["GarageYrBlt"].median()

        cols_to_fix = [
            "BsmtFinSF1",
            "BsmtFinSF2",
            "BsmtUnfSF",
            "TotalBsmtSF",
            "BsmtFullBath",
            "BsmtHalfBath",
            "GarageCars",
            "GarageArea",
        ]

        for col in cols_to_fix:
            self.bsmt_medians_[col] = X[col].median()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X["LotFrontage"] = X["LotFrontage"].fillna(
            X["Neighborhood"].map(self.lotfrontage_neighborhood_median_)
        )
        X["LotFrontage"] = X["LotFrontage"].fillna(
            self.lotfrontage_neighborhood_median_.median()
        )

        X["MasVnrArea"] = X["MasVnrArea"].fillna(self.masvnrarea_median_)
        X["GarageYrBlt"] = X["GarageYrBlt"].fillna(self.garageyrbult_median_)

        X["GarageAge"] = X["YrSold"] - X["GarageYrBlt"]
        X["TotalArea"] = (
            X["1stFlrSF"]
            + X["2ndFlrSF"]
            + X["TotalBsmtSF"]
            + X["GrLivArea"]
            + X["GarageArea"]
            + X["LotArea"]
        )
        X["TotalPorch"] = (
            X["OpenPorchSF"]
            + X["EnclosedPorch"]
            + X["3SsnPorch"]
            + X["ScreenPorch"]
        )
        X["Bath"] = (
            X["FullBath"]
            + 0.5 * X["HalfBath"]
            + X["BsmtFullBath"]
            + 0.5 * X["BsmtHalfBath"]
        )
        X["Age"] = X["YrSold"] - X["YearBuilt"]
        X["QualArea"] = X["OverallQual"] + X["GrLivArea"]
        X["RemodAge"] = X["YrSold"] - X["YearRemodAdd"]
        X["HasPool"] = (X["PoolArea"] > 0).astype(int)

        X["SaleCondition"] = X["SaleCondition"].replace({"Partial": "Normal"})
        rare_conditions = ["AdjLand", "Alloca", "Family"]
        X["SaleCondition"] = X["SaleCondition"].replace(rare_conditions, "Other")

        categorical_fill = {
            "PoolQC": "None",
            "MiscFeature": "None",
            "Alley": "None",
            "Fence": "None",
            "MasVnrType": "None",
            "Electrical": "None",
            "FireplaceQu": "None",
            "GarageFinish": "None",
            "GarageQual": "None",
            "GarageType": "None",
            "GarageCond": "None",
            "BsmtExposure": "None",
            "BsmtFinType2": "None",
            "BsmtFinType1": "None",
            "BsmtQual": "None",
            "BsmtCond": "None",
        }

        for col, fill_value in categorical_fill.items():
            X[col] = X[col].fillna(fill_value)

        for col, median_value in self.bsmt_medians_.items():
            X[col] = X[col].fillna(median_value)

        X["GarageAge"] = X["GarageAge"].fillna(X["GarageAge"].median())
        X["GarageAge"] = np.log1p(X["GarageAge"].clip(lower=0))

        X["LotArea"] = np.log1p(X["LotArea"].clip(lower=0))
        X["MiscVal"] = np.log1p(X["MiscVal"].clip(lower=0))
        X["Bath"] = np.log1p(X["Bath"].clip(lower=0))
        X["MSSubClass"] = np.log1p(X["MSSubClass"].clip(lower=0))
        X["MasVnrArea"] = np.sqrt(np.log1p(X["MasVnrArea"].clip(lower=0)))
        

        X.drop(columns=["GarageYrBlt"], inplace=True, errors="ignore")
        return X
