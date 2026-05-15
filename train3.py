import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
import joblib
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score,root_mean_squared_log_error


class Regressor:
    def __init__(self,X_train,X_test,y_train,y_test,prep1,prep2):
        self.X_train = X_train
        self.X_val = X_test
        self.y_train = y_train
        self.y_val = y_test
        self.preprocessor1 = prep1
        self.preprocessor2 = prep2



    def evaluate(self,model,name):
        model.fit(self.X_train,self.y_train)
        preds_log = model.predict(self.X_val)
        preds = np.expm1(preds_log)
        y_val_real = np.expm1(self.y_val)
        rmsle = root_mean_squared_log_error(y_val_real,preds)
        r2 = r2_score(y_val_real,preds)
        mse = mean_squared_error(y_val_real,preds)

        print(f"Model: {name}")
        print(f"RMSLE: {rmsle}")
        print(f"MSE: {mse}")
        print(f"R2 Score: {r2}")

        print("-----------------------")
        return model
    
    def get_xgboost_model(self):
        pipeline = Pipeline([
            ('preprocessor',self.preprocessor1),
            ('model',XGBRegressor(
                n_estimators = 1000,
                learning_rate = 0.03,
                max_depth = 3,
                subsample = 0.8
            ))
        ])

        return self.evaluate(pipeline,'XGBOOST')
    
    @staticmethod
    def feature_engineering(df):
        df = df[df['SaleCondition'] != 'Partial']

        # Feature Engineering
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
        df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].median())
        df['MasVnrArea'] = np.log1p(df['MasVnrArea'])
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].median())
        df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']
        df['GarageAge'] = np.log1p(df['GarageAge'])
        df['TotalArea'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF'] + df['GrLivArea'] + df['GarageArea'] + df['LotArea']
        df.drop('GarageYrBlt',axis = 1,inplace = True,errors='ignore')
        df['TotalPorch'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
        df['Bath'] = df['FullBath'] + 0.5*df['HalfBath'] + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']
        df['Age'] = df['YrSold'] - df['YearBuilt']
        df['QualArea'] = df['OverallQual'] + df['GrLivArea']
        df['LotArea']=np.log1p(df['LotArea'])
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
        df['Bath'] = np.log1p(df['Bath'])
        df['HasPool'] = (df['PoolArea'] >0).astype(int)
        df['MasVnrArea']=np.sqrt(df['MasVnrArea'])
        df['MiscVal']=np.log1p(df['MiscVal'])
        df['GarageAge'] = df['GarageAge'].fillna(df['GarageAge'].median())
        df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].mean())
        df['MSSubClass'] = np.log1p(df['MSSubClass'])
        df['SaleCondition'] = df['SaleCondition'].replace({'Partial': 'Normal'})
        rare_conditions = ['AdjLand', 'Alloca', 'Family']
        df['SaleCondition'] = df['SaleCondition'].replace(rare_conditions, 'Other')
        df['PoolQC'] = df['PoolQC'].fillna("None")
        df['MiscFeature'] = df['MiscFeature'].fillna("None")
        df['Alley'] = df['Alley'].fillna("None")
        df['Fence'] = df['Fence'].fillna("None")
        df['MasVnrType'] = df['MasVnrType'].fillna("None")
        df['Electrical'] = df['Electrical'].fillna("None")
        df['FireplaceQu'] = df['FireplaceQu'].fillna("None")
        df['GarageFinish'] = df['GarageFinish'].fillna("None")
        df['GarageQual'] = df['GarageQual'].fillna("None")
        df['GarageType'] = df['GarageType'].fillna("None")
        df['GarageCond'] = df['GarageCond'].fillna("None")
        df['BsmtExposure'] = df['BsmtExposure'].fillna("None")
        df['BsmtFinType2'] = df['BsmtFinType2'].fillna("None")
        df['BsmtFinType1'] = df['BsmtFinType1'].fillna("None")
        df['BsmtQual'] = df['BsmtQual'].fillna("None")
        df['BsmtCond'] = df['BsmtCond'].fillna("None")
        cols_to_fix = [
            'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
            'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea'
        ]

        for col in cols_to_fix:
            df[col] = df[col].fillna(df[col].median())

        df['TotalArea'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF'] + df['GrLivArea'] + df['GarageArea'] + df['LotArea']
        df['Bath'] = df['FullBath'] + 0.5*df['HalfBath'] + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']
        df['Bath'] = np.log1p(df['Bath']) 

        X = df.drop('SalePrice',axis = 1)
        y = df['SalePrice']
        y = np.log1p(y)

        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        X_train, X_val, y_train , y_val = train_test_split(X,y,test_size=0.2,random_state=42)


        num_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])

        preprocessor2 = ColumnTransformer(
            transformers=[
                ('numeric',num_pipeline,num_cols),
                ('cat', OneHotEncoder(drop = 'first',handle_unknown='ignore', sparse_output=False), cat_cols)
            ],remainder='passthrough')

        preprocessor1 = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
            ],remainder='passthrough')
        
        return X_train, X_val, y_train, y_val, preprocessor1 , preprocessor2

        
    
    def get_ridge_regression(self):
        pipeline = Pipeline([
            ('preprocessor',self.preprocessor2),
            ('model',RidgeCV(alphas=[0.1, 1.0, 5.0, 10.0, 50.0, 100.0], cv=5))
        ])

        return self.evaluate(pipeline,'Ridge')
    
    def get_lgbm_regressor(self):
        pipeline = Pipeline([
            ('preprocessor',self.preprocessor1),
            ('model', LGBMRegressor(random_state= 42,
                                                   n_estimators = 200, 
                                                   max_depth=20,
                                                   learning_rate = 0.1,
                                                   num_leaves=31,
                                                   ))
        ])

        

        return self.evaluate(pipeline,'LGBM Regressor')
    
       


file_path = "/home/amritkg9009/Downloads/house-prices-advanced-regression-techniques/train.csv"
df = pd.read_csv(file_path)


X_train , X_test, y_train , y_test, prep1, prep2 = Regressor.feature_engineering(df)

reg = Regressor(
    X_train,
    X_test,
    y_train,
    y_test,
    prep1,
    prep2
    )

xgb_model = reg.get_xgboost_model()
ridge_model = reg.get_ridge_regression()
lgbm_model = reg.get_lgbm_regressor()

w = [1,3,1]
er = VotingRegressor([('xgb',xgb_model),('ridge',ridge_model),('lgbm',lgbm_model)],
                     weights=w
                     )

reg.evaluate(er,"VOTING_ENSEMBLE")

joblib.dump(
    er,
    "house_price_model.pkl"
)

joblib.dump(
    X_train.columns.tolist(),
    "model_features.pkl"
)

print("✅ House Price Model Saved Successfully!")