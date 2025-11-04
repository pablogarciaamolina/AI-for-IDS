import os
import numpy as np
import logging
import joblib
import pandas as pd

from tabstar.tabstar_model import TabSTARClassifier

from ._base import SklearnTrainableModel
from src.models.config import TABSTAR_CONFIG, TABSTAR_PARAMS, TABSTAR_SAVING_PATH


class TabSTARModel(SklearnTrainableModel):

    model: TabSTARClassifier

    def __init__(self, name: str = "tabstar"):

        model = TabSTARClassifier(**TABSTAR_CONFIG)
        super().__init__(name, model)


    def fit(self, x_train: pd.DataFrame | np.ndarray, y_train: pd.Series | np.ndarray) -> None:
        """
        Performs fit method

        Args:
            x_train: Training data features
            y_train: Training data labels
        """

        if type(x_train) == np.ndarray and type(y_train) == np.ndarray:
            assert len(x_train.shape) == 2, "Unable to convert x_train to DataFrame"
            x_train = pd.DataFrame(x_train, columns=[f"Feature {i+1}" for i in range(x_train.shape[1])])
            assert len(x_train.shape) == 2, "Unable to convert y_train to Series"
            y_train = pd.Series(y_train, name="Label")
        else:
            assert type(x_train) == pd.DataFrame, "x_train in not DataFrame or NDArray"
            assert type(y_train) == pd.Series, "y_train in not Series or NDArray"

        logging.info("Fitting TabSTAR model...")
                
        self.model.fit(
            x_train,
            y_train
        )


    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predicts labels for the given inputs, using batches.

        Args:
            x: Inputs for which to predict the labels
        """

        logging.info("TabSTAR predicting labels...")

        batch_size = x.shape[0] if TABSTAR_PARAMS["predicting_batch_size"] == -1 else TABSTAR_PARAMS["predicting_batch_size"]
        assert isinstance(batch_size, int) and batch_size > 0

        if type(x) == np.ndarray:
            assert len(x.shape) == 2, "Unable to convert x_train to DataFrame"
            x: pd.DataFrame = pd.DataFrame(x, columns=[f"Feature {i+1}" for i in range(x.shape[1])])
        else:
            assert type(x) == pd.DataFrame, "x is not DataFrame object"

        logging.info(f"Starting batch prediction with batch size of {min(batch_size, x.shape[0])}... (Total of {x.shape[0]} samples to predict)")
        preds = []
        for i in range(0, x.shape[0], batch_size):
            predictions = self.model.predict(x.iloc[i:i+batch_size])
            preds.append(predictions)
            logging.info(f"{min(i + batch_size, x.shape[0])} samples predicted")

        predictions = np.concatenate(preds)

        return self.model.preprocessor_.target_transformer.inverse_transform(predictions)

    def save(self):
        
        logging.info(f"Saving model...")

        filepath = os.path.join(TABSTAR_SAVING_PATH, self.name + ".zip")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        self.model.save(filepath)
        logging.info(f"Model and components saved to {filepath}")
    
    def load(self):
        
        logging.info(f'Loading model...')

        filepath = os.path.join(TABSTAR_SAVING_PATH, self.name + ".zip")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file '{filepath}' not found.")

        self.model = TabSTARClassifier.load(filepath)

        if hasattr(self.model, "device"):
            self.model.device = "auto"

    