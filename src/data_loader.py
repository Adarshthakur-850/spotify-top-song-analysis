import pandas as pd
from .config import DATA_URL

class DataLoader:
    def load_data(self):
        print("Loading data from URL...")
        df = pd.read_csv(DATA_URL)
        return df

    def clean_data(self, df):
        print("Cleaning data...")
        df = df.dropna()
        df = df.drop_duplicates()
        
        if 'track_name' in df.columns:
             df = df.drop_duplicates(subset=['track_name', 'artist_name'])
             
        return df
