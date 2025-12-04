import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        # We don't strictly need scaling for Random Forest, but it's good practice
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def load_and_process(self):
        df = pd.read_csv(self.file_path)
        df = df.set_index('Datetime')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index() # Ensure time is sorted
        
        # Normalize
        data_normalized = self.scaler.fit_transform(df[['PJME_MW']])
        return data_normalized, self.scaler

    def create_sequences(self, data, seq_length=24):
        # PAST 24 hours -> PREDICT next 1 hour
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length].flatten()) # Flatten to 1D array
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)