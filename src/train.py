from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import numpy as np
from .config import MODELS_DIR, TEST_SIZE, RANDOM_STATE

class Trainer:
    def split_data(self, X, y):
        return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    def train_model(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        return y_pred, mae, rmse, r2

    def save_model(self, model, filename):
        path = os.path.join(MODELS_DIR, filename)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved model: {path}")
