from abc import ABC, abstractmethod
from typing import Any

from src.models._base import BaseModel


class TrainableModel(BaseModel):

    model: Any

    def __init__(self, name: str, model: Any):
        
        super().__init__(name)
        self.model = model

    @abstractmethod
    def save(self) -> None:
        """
        Saves the trained model.
        """

        pass

    @abstractmethod
    def load(self) -> None:
        """
        Loads the trained model.
        """

        pass

class SklearnTrainableModel(TrainableModel):

    def __init__(self, name: str, model: Any):

        super().__init__(name, model)

    @abstractmethod
    def fit(self, x, y) -> None:
        """
        X: Training data features
        y: Training data labels
        """

        pass
        