import os
import numpy as np
import logging
import joblib

from sklearn.preprocessing import LabelEncoder
from tabicl import TabICLClassifier

from ._base import SklearnTrainableModel
from src.models.config import TABICL_CONFIG, TABICL_PARAMS, TABICL_SAVING_PATH


class TabICLModel(SklearnTrainableModel):

    model: TabICLClassifier

    def __init__(self, name: str = "tabicl"):

        model = TabICLClassifier(**TABICL_CONFIG)
        super().__init__(name, model)

        self.label_encoder = LabelEncoder()


    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Performs .fit method

        Args:
            x_train: Training data features
            y_train: Training datat labels
        """

        logging.info("Fitting TabICL model...")

        # y_train = self.label_encoder.fit_transform(y_train)
        
        self.model.fit(
            x_train,
            y_train
        )


    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts labels for the given inputs, using batches.

        Args:
            x: Iputs for which to predict the labels
        """

        logging.info("TabICL predicting labels...")

        batch_size = x.shape[0] if TABICL_PARAMS["predicting_batch_size"] == -1 else TABICL_PARAMS["predicting_batch_size"]
        assert isinstance(batch_size, int) and batch_size > 0

        logging.info(f"Starting batch prediction with batch size of {min(batch_size, x.shape[0])}... (Total of {x.shape[0]} samples to predict)")
        preds = []
        for i in range(0, x.shape[0], batch_size):
            predictions = self.model.predict(x[i:i+batch_size]) 
            preds.append(predictions)
            logging.info(f"{min(i + batch_size, x.shape[0])} samples predicted")

        predictions = np.concatenate(preds)

        # return self.label_encoder.inverse_transform(predictions)
        return predictions

    def save(self):
        
        logging.info(f"Saving model...")

        filepath = os.path.join(TABICL_SAVING_PATH, self.name + ".zip")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        save_dict = {
            "model": self.model,
            # "label_encoder": self.label_encoder,
        }

        joblib.dump(save_dict, filepath)
        logging.info(f"Model and components saved to {filepath}")

    
    def load(self):
        
        logging.info(f'Loading model...')

        filepath = os.path.join(TABICL_SAVING_PATH, self.name + ".zip")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file '{filepath}' not found.")

        save_dict = joblib.load(filepath)
        self.model = save_dict["model"]
        # self.label_encoder = save_dict["label_encoder"]

        if hasattr(self.model, "device"):
            self.model.device = "auto"

    