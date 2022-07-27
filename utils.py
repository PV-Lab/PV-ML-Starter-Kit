import pandas as pd
import numpy as np

def normalize(df):
    df = df.copy()
    for col in range(df.shape[1] - 1): # exclude last column
        df.iloc[:,col] = (df.iloc[:,col] - np.min(df.iloc[:,col])) / (np.max(df.iloc[:,col]) - np.min(df.iloc[:,col]))
    return df

def unnormalize(df_raw, df_norm):
    df_norm = df_norm.copy()
    for col in range(df_norm.shape[1] - 1): # exclude last column
        df_norm.iloc[:,col] = df_norm.iloc[:,col] * (np.max(df_raw.iloc[:,col]) - np.min(df_raw.iloc[:,col])) + np.min(df_raw.iloc[:,col])
    return df_norm