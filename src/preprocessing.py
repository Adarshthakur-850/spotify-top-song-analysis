import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def encode_categorical(self, df, columns):
        print("Encoding categorical features...")
        for col in columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        return df

    def create_features(self, df):
        print("Creating audio features...")
        if 'energy' in df.columns and 'danceability' in df.columns:
            df['energy_dance_ratio'] = df['energy'] / (df['danceability'] + 0.01)
        return df
    
    def scale_features(self, X_train, X_test):
        print("Scaling numerical features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
