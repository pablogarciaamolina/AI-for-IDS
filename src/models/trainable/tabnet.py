import os
from typing import Optional
import logging
import numpy as np
import matplotlib.pyplot as plt

from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer

from ._base import SklearnTrainableModel
from src.models.config import TABNET_PRETRAINER_CONFIG, TABNET_CONFIG, TABNET_PRETRAINING_PARAMS, TABNET_TRAINING_PARAMS, TABNET_SAVING_PATH

class TabNetModel(SklearnTrainableModel):

    model: TabNetClassifier

    def __init__(self, name: str = "tabnet", pretrain: bool = True) -> None:
        """
        Constructor for the class

        Args:
            pretrain: Whether to perform, pretraining or not
            name: Name for the model
        """

        model = TabNetClassifier(**TABNET_CONFIG)
        super().__init__(name, model)
        self.pretrainer = TabNetPretrainer(**TABNET_PRETRAINER_CONFIG) if pretrain else None

    def fit(self, x: np.ndarray, y: np.ndarray, x_val: Optional[np.ndarray], y_val: Optional[np.ndarray]) -> None:
        """
        Performs pretreining (if specified) and then trains the model based on the configuration paramenters

        Args:
            x_train: Training data features
            y_train: Training datat labels
            x_val: Optional validation data features
            y_val: Optional validation data labels
            augmentation: Whether to perform data augmentation or not
        """

        if self.pretrainer:
            logging.info("Pretraining for TabNet...")
            eval_set = [x_val] if x_val is not None else None
            self.pretrainer.fit(
                x,
                eval_set=eval_set,
                **TABNET_PRETRAINING_PARAMS
            )

        logging.info("Training TabNet model...")
        if (x_val is not None) and (y_val is not None):
            eval_set = [(x_val, y_val),]
            eval_name = ['val']
        else:
            eval_set = None
            eval_name = None
        self.model.fit(
            x, y,
            eval_set=eval_set,
            eval_name=eval_name,
            **TABNET_TRAINING_PARAMS,
            from_unsupervised=self.pretrainer,
        )
    
    def predict(self, x) -> np.ndarray:

        return self.model.predict(x)

    def plot_metrics(self) -> None:
        """
        Plots the metrics for the last training done to the model.
        NOTE: Not having trained the model will raise an error. Loading the model from memory does not include its past training history.
        """
        
        plt.plot(self.model.history['lr'])

        clf = self.model
        train_loss = clf.history["loss"]
        if "val_accuracy" in clf.history.history.keys():
            val_metric = clf.history["val_accuracy"]
        else:
            val_metric = None

        # Plot Loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label="Train Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Tabnet Training Loss Over Epochs")
        plt.legend()

        # Plot Evaluation Metric
        plt.subplot(1, 2, 2)
        if val_metric is not None:
            plt.plot(val_metric, label="Validation Accuracy", color="orange")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Tabnet Validation Acuracy Over Epochs")
        plt.legend()

        plt.show()

    def save(self) -> None:

        path = os.path.join(TABNET_SAVING_PATH, self.name)
        self.model.save_model(path)

    def load(self) -> None:

        path = os.path.join(TABNET_SAVING_PATH, self.name + ".zip")
        self.model.load_model(path)

    def explain(self, x) -> None:
        """
        Local explanation method for visualizing generated masks over a features set

        Args:
            x: Data features to use for explanation
        """

        explanations, masks = self.model.explain(x)

        fig, axs = plt.subplots(1, 3, figsize=(20,20))
        for i in range(3):
            axs[i].imshow(masks[i][:50])
            axs[i].set_title(f"mask {i}")