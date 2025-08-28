import numpy as np
from collections import Counter

import pandas as pd

from src.data import smote
from src.models.trainable._base import SklearnTrainableModel
from src.models.API.llm import LLModel, Mistral, Gemini
from src.models.trainable import TabNetModel, TabPFNModel, MLClassifier
from src.pipelines._base import BasePipeline


class TTPipeline(BasePipeline):
    """
    Tabular-Trainable Pipeline
    """

    model: SklearnTrainableModel

    def __init__(self, model: SklearnTrainableModel):
        super().__init__(model)

    def train(self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val = None,
        y_val = None,
        cv: int = 10,
        smote_augmentation: bool = False
    ) -> None:
        
        if smote_augmentation:
            x_train, y_train = smote(
                x_train,
                y_train,
            )

        if isinstance(self.model, TabNetModel):
            self.model.fit(x_train, y_train, x_val, y_val)
        elif isinstance(self.model, MLClassifier):
            self.model.fit(x_train, y_train, cv=cv, verbose=1)
        else:
            self.model.fit(x_train, y_train)

    def evaluate(self, x_test, y_test, other_info: str = None) -> dict:
        """
        Pipeline for avaluating the model

        x_test: Test data to be classified.
        y_test: True labels.
        other_info: Other information to include in the report.
        """
        
        pred = self.model.predict(x_test)

        results = self.evaluate_given_predictions(
            pred,
            y_test,
            other_info
        )

        return results


class TAPipeline(BasePipeline):
    """
    Tabular-API Pipeline with Majority Voting
    """

    model: LLModel

    def __init__(self, model: LLModel):
        super().__init__(model)
    
    def majority_vote(
        self,
        predictions: list[list[str]],
        unkown_label = None
    ) -> list[str]:
        """
        Computes the majority vote for each sample.
        
        Args:
            predictions: A list of lists, each containing multiple predictions per sample.
        
        Returns:
            A list containing the final prediction per sample based on majority voting.
        """

        final_predictions = []
        for sample_preds in predictions:
            filtered_preds = [pred for pred in sample_preds if pred is not None]
            if filtered_preds:
                final_predictions.append(Counter(filtered_preds).most_common(1)[0][0])
            else:
                final_predictions.append(unkown_label)
        return final_predictions

    def evaluate(self,
        icl_data: pd.DataFrame,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        task: str = "Behaving like an Intrusion Detection System, classifying the data based on the PC features in the possible classes of attacks or not attack",
        class_column: str = "Label",
        num_predictions: int = 3,
        other_info: str = None
    ) -> dict:
        """
        Evaluates the model with majority voting.
        
        Args:
            icl_data: A pandas DataFrame containing the ICL examples.
            x_test: A pandas DataFrame containing the test data to be classified.
            y_test: A pandas Series containing the true labels.
            task: A sentence that further explains the classification task.
            class_column: The name of the column containing the labels in `icl_data`.
            num_predictions: Number of times to generate predictions for majority voting.
            other_info: Other information to include in the report
        
        Returns:
            A dictionary containing the metrics and a figure with the report.
        """

        if isinstance(self.model, Mistral):
            config = {
                "n": num_predictions
            }
        elif isinstance(self.model, Gemini):
            config = {
                "candidate_count": num_predictions
            }

        preds = self.model.predict(icl_data, x_test, task, class_column, **config)
        
        final_preds = self.majority_vote(preds)

        if other_info:
            other_info = other_info + f", majority voting with {num_predictions} predictions"
        else:
            other_info = f"majority voting with {num_predictions} predictions"

        results = self.evaluate_given_predictions(
            final_preds,
            y_test,
            other_info
        )

        return results
