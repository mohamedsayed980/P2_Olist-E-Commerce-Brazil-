# utils.py
import numpy as np

def get_numeric_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_cat_cols(df):
    return df.select_dtypes(include=["object", "category"]).columns.tolist()
