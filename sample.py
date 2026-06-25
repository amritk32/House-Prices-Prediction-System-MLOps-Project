import pandas as pd
pd.set_option("display.max_rows",None)
import joblib

template = joblib.load("default_template.pkl")

input_df = template.copy()

print(input_df.dtypes)