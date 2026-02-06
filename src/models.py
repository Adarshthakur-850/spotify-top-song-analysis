from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class ModelFactory:
    @staticmethod
    def get_model(model_type, **kwargs):
        if model_type == 'linear_regression':
            return LinearRegression(**kwargs)
        elif model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=100, random_state=42, **kwargs)
        elif model_type == 'xgboost':
            return XGBRegressor(n_estimators=100, random_state=42, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
