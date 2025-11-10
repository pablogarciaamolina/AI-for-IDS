import os
import numpy as np
import logging

import joblib
from tabdpt import TabDPTClassifier

from ._base import SklearnTrainableModel
from src.models.config import TABDPT_CONFIG, TABDPT_SAVING_PATH


class TabDPTModel(SklearnTrainableModel):

    model: TabDPTClassifier

    def __init__(self, name: str = "tabdpt"):

        model = TabDPTClassifier(**TABDPT_CONFIG)
        super().__init__(name, model)


    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Performs fit method

        Args:
            x_train: Training data features
            y_train: Training data labels
        """

        logging.info("Fitting TabDPT model...")
                
        self.model.fit(
            x_train,
            y_train
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts labels for the given inputs.

        Args:
            x: Inputs for which to predict the labels
        """

        logging.info("TabDPT predicting labels...")

        return self.model.predict(x)

    def save(self):
        
        logging.info(f"Saving model...")

        filepath = os.path.join(TABDPT_SAVING_PATH, self.name + ".zip")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        save_dict = {
            "model": self.model,
        }

        joblib.dump(save_dict, filepath)
        logging.info(f"Model and components saved to {filepath}")
    
    def load(self):
        
        logging.info(f'Loading model...')

        filepath = os.path.join(TABDPT_SAVING_PATH, self.name + ".zip")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file '{filepath}' not found.")

        save_dict = joblib.load(filepath)
        self.model = save_dict["model"]

        if hasattr(self.model, "device"):
            self.model.device = None

    