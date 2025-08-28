# src/data/base.py
from abc import ABC, abstractmethod
import os
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.utils import balanced_sample

class BaseDataset(ABC):
    """
    Abstract base class defining a common interface for all datasets.
    """

    def __init__(self):
        """
        Args:
            source (str): The data source (e.g., file or folder path).
        """
        # self.source = os.path.abspath(source)
        self.data = None

    @abstractmethod
    def _download(self) -> None:
        """
        Downloads data from the source if needed.
        """
        pass

    @abstractmethod
    def _collect(self) -> None:
        """
        Retrieves raw data from the source (saves it in memory).
        """
        pass

    @abstractmethod
    def _preprocess(self) -> None:
        """
        Apply preprocessing steps to the data.
        """
        pass

    @abstractmethod
    def _save(self) -> None:
        """
        Saves stored data for future loading.
        """
        pass

    @abstractmethod
    def load(self) -> None:
        """
        Loads the prepared data. Check if data is saved on the corresponding directory, else completes the hole pipeline for loading the dataset.
        """
        pass

class TabularDataset(BaseDataset):
    """
    Base class for handling tabular datasets using pandas.
    """

    def __init__(self, data_dir: str, file_name: str, url: str = None):
        """
        Args:
            data_dir (str): Directory where the dataset is stored.
            file_name (str): Name of the CSV file.
            url (str, optional): URL for downloading the dataset if not available.
        """
        super().__init__()
        self.data_dir = os.path.abspath(data_dir)
        self.file_name = file_name
        self.file_path = os.path.join(self.data_dir, self.file_name)
        self.url = url

    def _save(self) -> None:
        """
        Saves the processed dataset to a CSV file.
        """
        try:
            if self.data is not None:
                self.data.to_csv(self.file_path, index=False)
                logging.info(f"Data saved to {self.file_path}")
        except Exception as e:
            logging.error(f"Failed to save data: {e}")

    def load(self):
        """
        Loads the dataset. If a saved version exists, loads it; otherwise, runs the full pipeline.
        """
        logging.info("Loading dataset...")
        if os.path.isfile(self.file_path):
            self._collect()
        else:
            self._collect()
            self._preprocess()
            self._save()

    def balance_(self, n: int, category_col: str = "Label") -> None:
        """
        Inplace method for balancing the data based on a categorical variable.

        Args:
            n: Number of samples per category
            category_col: Column name of the categorical variable
        """

        self.data = balanced_sample(self.data, category_col=category_col, n_per_class=n)

    def train_test_split(self, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Draws train and test subsets splited randomly from the data

        Args:
            test_size: Float representing the proportion of the dataset to include in the test split

        Returns:
            train data and test data DataFrames
        """

        return train_test_split(self.data, test_size=test_size)

