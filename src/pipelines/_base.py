from abc import ABC, abstractmethod
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

from src.models._base import BaseModel


class BasePipeline(ABC):

    def __init__(self, model: BaseModel):
        
        self.model = model

    @abstractmethod
    def evaluate(self, x_test, y_test) -> dict:
        """
        Method for evaluating a classifier

        Args:
            x_test: Testing features
            y_test: Testing labels

        Returns:
            A dictionary containing the metrics and a figure with the report.
        """

        pass

    
    def evaluate_given_predictions(self,
        predictions,
        true_labels,
        other_info: str = None
    ) -> dict:
        """
        Evaluating with predictions already obtained

        Args:
            predictions: Predictions
            true_labels: True labels
            other_info: Other information to include in the report

        Returns:
            A dictionary containing the metrics and a figure with the report.
        
        """
        
        classes = np.sort(np.unique(true_labels))
        metrics = classification_report(y_true=true_labels, y_pred=predictions, target_names=classes, output_dict=True)
        precision = [metrics[target_name]['precision'] for target_name in classes]
        recall = [metrics[target_name]['recall'] for target_name in classes]
        f1_score = [metrics[target_name]['f1-score'] for target_name in classes]
        data = np.array([precision, recall, f1_score])

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(data, cmap='Pastel1', annot=True, fmt='.2f', xticklabels=classes, yticklabels=['Precision', 'Recall', 'F1-score'], ax=ax, annot_kws={"color": "black"})
        ax.set_title(f'Metrics Report ({self.model.name}{" - " + other_info if other_info is not None else ""})')
        fig.tight_layout()

        accuracy = accuracy_score(true_labels, predictions)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "metrics_report": fig
        }