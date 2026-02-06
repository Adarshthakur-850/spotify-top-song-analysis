from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.visualization import Visualizer
from src.models import ModelFactory
from src.train import Trainer
import pandas as pd

def main():
    print("=== Spotify Top Songs Analysis ===")
    
    loader = DataLoader()
    df = loader.load_data()
    df = loader.clean_data(df)
    print(f"Dataset Shape: {df.shape}")
    
    preprocessor = Preprocessor()
    feature_cols = [
        'acousticness', 'danceability', 'energy', 'instrumentalness', 
        'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
    ]
    target = 'popularity'
    
    available_features = [c for c in feature_cols if c in df.columns]
    
    df = preprocessor.create_features(df)
    if 'energy_dance_ratio' in df.columns:
        available_features.append('energy_dance_ratio')
        
    print(f"Features: {available_features}")
    
    viz = Visualizer()
    viz.plot_distributions(df)
    viz.plot_correlations(df)
    
    X = df[available_features]
    y = df[target]
    
    trainer = Trainer()
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    models = ['linear_regression', 'random_forest', 'xgboost']
    best_r2 = -float('inf')
    best_model = None
    
    for m_name in models:
        print(f"\n--- Training {m_name} ---")
        model = ModelFactory.get_model(m_name)
        model = trainer.train_model(model, X_train_scaled, y_train)
        y_pred, mae, rmse, r2 = trainer.evaluate_model(model, X_test_scaled, y_test)
        
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.4f}")
        
        viz.plot_actual_vs_predicted(y_test, y_pred, m_name)
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            
    if best_model:
        trainer.save_model(best_model, 'best_model.pkl')
        
    print("\n=== Pipeline Completed ===")

if __name__ == "__main__":
    main()
