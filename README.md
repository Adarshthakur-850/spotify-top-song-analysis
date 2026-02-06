# Spotify Top Songs Analysis and Popularity Prediction

An end-to-end data science project extracting insights from Spotify's top songs dataset and predicting track popularity.

## Project Structure
```
spotify top song analysis/
├── models/             # Saved ML models (.pkl)
├── plots/              # Generated visualizations
├── src/
│   ├── config.py       # Configuration constants
│   ├── data_loader.py  # Data fetching and cleaning
│   ├── preprocessing.py# Scaling and feature engineering
│   ├── visualization.py# Plotting functions (Correlation, Dist)
│   ├── models.py       # Model factory (Linear, RF, XGBoost)
│   ├── train.py        # Training loop and metric evaluation
├── main.py             # Entry point
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## Features
- **Data Analysis**: Distribution of popularity, energy, danceability.
- **Visualizations**: Correlation heatmaps, feature trends.
- **Machine Learning**:
  - Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor
- **Feature Engineering**: Creating ratios like Energy/Danceability.
- **Evaluation**: RMSE, MAE, R2 scores.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the analysis pipeline:
```bash
python main.py
```

## Output
- **Plots**: Check `plots/` folder.
- **Model**: Best performing model saved in `models/best_model.pkl`.
  

