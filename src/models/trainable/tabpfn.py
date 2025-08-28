import os
import numpy as np
import logging
import joblib

from sklearn.preprocessing import LabelEncoder
from tabpfn import TabPFNClassifier
from tabpfn.config import ModelInterfaceConfig
from tabpfn_extensions.many_class import ManyClassClassifier

from ._base import SklearnTrainableModel
from src.models.config import TABPFN_CONFIG, TABPFN_SAVING_PATH, TABPFN_EXPERT_CONFIG, TABPFN_PARAMS, TABPFN_MANY_CLASS_CONFIG


class TabPFNModel(SklearnTrainableModel):

    def __init__(self, name: str = "tabpnf"):

        expert_config = ModelInterfaceConfig(
            **TABPFN_EXPERT_CONFIG
        )
        model = TabPFNClassifier(**TABPFN_CONFIG, inference_config=expert_config)
        super().__init__(name, model)

        self.extension = None
        self.label_encoder = LabelEncoder()


    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Performs .fit method

        Args:
            x_train: Training data features
            y_train: Training datat labels
        """

        logging.info("Fitting TabPNF model...")

        if len(np.unique(y_train)) > TABPFN_EXPERT_CONFIG["MAX_NUMBER_OF_CLASSES"]:
            self.extension = ManyClassClassifier(
                self.model,
                **TABPFN_MANY_CLASS_CONFIG
            )

        y_train = self.label_encoder.fit_transform(y_train)
        
        if self.extension is None:
            logging.info("Training base TabPFN model")
            self.model.fit(
                x_train,
                y_train
            )
        else:
            logging.info("Training ManyClassClassifier extension")
            self.extension.fit(
                x_train,
                y_train
            )


    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts labels for the given inputs, using batches.

        Args:
            x: Iputs for which to predict the labels
        """

        logging.info("TabPFN predicting labels...")

        batch_size = x.shape[0] if TABPFN_PARAMS["predicting_batch_size"] == -1 else TABPFN_PARAMS["predicting_batch_size"]
        assert isinstance(batch_size, int) and batch_size > 0

        logging.info(f"Starting batch prediction with batch size of {min(batch_size, x.shape[0])}... (Total of {x.shape[0]} samples to predict)")
        preds = []
        for i in range(0, x.shape[0], batch_size):
            if self.extension is None:
                predictions = self.model.predict(x[i:i+batch_size]) 
            else:
                predictions = self.extension.predict_proba(x[i:i+batch_size])
                predictions = self.extension.classes_[np.argmax(predictions, axis=1)]

            preds.append(predictions)
            logging.info(f"{min(i + batch_size, x.shape[0])} samples predicted")

        predictions = np.concatenate(preds)

        return self.label_encoder.inverse_transform(predictions)
    

    def save(self):
        
        logging.info(f"Saving model...")

        filepath = os.path.join(TABPFN_SAVING_PATH, self.name + ".zip")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        save_dict = {
            "model": self.model,
            "extension": self.extension,
            "label_encoder": self.label_encoder,
        }

        joblib.dump(save_dict, filepath)
        logging.info(f"Model and components saved to {filepath}")

    
    def load(self):
        
        logging.info(f'Loading model...')

        filepath = os.path.join(TABPFN_SAVING_PATH, self.name + ".zip")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file '{filepath}' not found.")

        save_dict = joblib.load(filepath)
        self.model = save_dict["model"]
        self.extension = save_dict["extension"]
        self.label_encoder = save_dict["label_encoder"]

        if hasattr(self.model, "device"):
            self.model.device = "auto"

    