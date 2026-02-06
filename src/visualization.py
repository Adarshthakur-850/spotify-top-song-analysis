import matplotlib.pyplot as plt
import seaborn as sns
import os
from .config import PLOTS_DIR

class Visualizer:
    def __init__(self):
        sns.set(style="whitegrid")

    def save_plot(self, fig, filename):
        path = os.path.join(PLOTS_DIR, filename)
        fig.savefig(path)
        plt.close(fig)
        print(f"Saved plot: {path}")

    def plot_distributions(self, df):
        plt.figure(figsize=(10, 6))
        sns.histplot(df['popularity'], kde=True, bins=30)
        plt.title('Popularity Distribution')
        self.save_plot(plt.gcf(), 'popularity_dist.png')

    def plot_correlations(self, df):
        plt.figure(figsize=(12, 10))
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        self.save_plot(plt.gcf(), 'correlation_heatmap.png')

    def plot_actual_vs_predicted(self, y_test, y_pred, model_name):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Popularity')
        plt.ylabel('Predicted Popularity')
        plt.title(f'{model_name}: Actual vs Predicted')
        self.save_plot(plt.gcf(), f'{model_name}_prediction.png')
