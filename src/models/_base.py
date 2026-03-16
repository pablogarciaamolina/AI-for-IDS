import os
from abc import ABC, abstractmethod
from typing import Any
import logging

from huggingface_hub import HfFolder, login

from src.models.config import HF_TOKEN

class Hugging_Face_Login:
    """
    Class for authentication with Hugging Face Hub once per process.
    """

    _logged_in = False

    @classmethod
    def login(cls):

        if cls._logged_in:
            return
        
        if not HF_TOKEN:
            logging.info(f'Hugging Face token not set. The model you are trying to use requires HF loging, please set the env variable "HF_TOKEN" or change its value in the config file')

        HfFolder.save_token(HF_TOKEN)

        login(token=HF_TOKEN)

        cls._logged_in = True
    

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