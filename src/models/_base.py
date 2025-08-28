from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):

    def __init__(self, name: str):
        
        self.name = name

    @abstractmethod
    def predict(self, X) -> Any:
        """
        Method for predicting samples with the model
    
        Args:
            X: The data for which we want to get the predictions.

        Returns:
            Class labels for each sample.
        """

        pass