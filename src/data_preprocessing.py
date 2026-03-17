import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    columns = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
    df = pd.read_csv(file_path, header=None, names=columns)

    df.drop(columns=['id'], inplace=True)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.to_numpy()