import os
import requests
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

from src.data.config import DATA_DIR, CICUNSW_URL, CICUSNW_DEFAULT_CONFIG
from src.data._base import TabularDataset
from src.data.utils import features_correction

class CIC_UNSW(TabularDataset):

    dataset_folder = "CIC-UNSW"

    def __init__(self, **kwargs):
        
        self.config = CICUSNW_DEFAULT_CONFIG.copy()
        self.config.update(kwargs)

        self.raw_data_dir = os.path.abspath(os.path.join(DATA_DIR, self.dataset_folder, "raw"))
        self.processed_data_dir = os.path.abspath(os.path.join(DATA_DIR, self.dataset_folder, "processed"))
        url = CICUNSW_URL

        super().__init__(self.processed_data_dir, self._get_id() + ".csv", url)

    def _get_id(self) -> str:
        
        sorted_values = sorted(self.config.items())
        return "cic_unsw_" + '_'.join([f"{key}({value})" for key, value in sorted_values])

    def _download(self) -> None:
        """
        Dowloads data files.
        """

        if not self.url:
            logging.error("No URL provided for dataset download.")
            return
        
        os.makedirs(self.raw_data_dir, exist_ok=True)
        csv_path = os.path.join(self.raw_data_dir, os.path.basename(self.url))

        logging.info(f"Downloading dataset from {self.url}...")
        response = requests.get(self.url, stream=True)
        if response.status_code == 200:
            with open(csv_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            os.makedirs(self.processed_data_dir, exist_ok=True)
        else:
            logging.error(f"Failed to download dataset from {self.url}.")

    def _collect(self):
        
        if not os.path.exists(self.processed_data_dir):
            self._download()
    
        if not os.path.isfile(self.file_path):
            logging.info(f"{self.file_path} not found. Collecting and cleaning raw data in {self.raw_data_dir}...")
            self.data = self._clean()
            self.data.to_csv(self.file_path, sep=",", index=False)
        else:
            self.data = pd.read_csv(self.file_path)

    def _clean(self) -> pd.DataFrame:
        """
        Method for cleaning the data.
        """

        logging.info("Cleaning raw data...")

        csv_files = [f for f in os.listdir(self.raw_data_dir) if f.endswith('.csv')]
        assert len(csv_files) == 1, f"Expected only one raw data file, got {len(csv_files)}"

        csv_path = os.path.join(self.raw_data_dir, csv_files[0])
        df = pd.read_csv(csv_path)

        # Drop non-numeric columns, keeping the labels (drop: Flow ID, Src IP, Dst IP, Timestamp)
        labels = df["Label"]
        df = df.select_dtypes(include=["number"])
        df["Label"] = labels

        return df

    def _preprocess(self):
        """
        Applies dataset-specific preprocessing, such as feature correction and cleaning.
        """

        logging.info("Starting preprocessing...")

        # Correct feature names
        features_correction(self.data)
        logging.info("Feature names corrected.")

        # QUANTIZATION
        logging.info("Starting quantization...")
        for col in self.data.columns:
            col_type = self.data[col].dtype
            if col_type != object:
                c_min = self.data[col].min()
                c_max = self.data[col].max()
                # Downcasting float64 to float32
                if str(col_type).find('float') >= 0 and c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    try:
                        self.data[col] = self.data[col].astype(np.float32)
                    except ValueError:
                        logging.warning(f"Failed to downcast {col} to float32")

                # Downcasting int64 to int32
                elif str(col_type).find('int') >= 0 and c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    self.data[col] = self.data[col].astype(np.int32)
        logging.info("Quantization complete.")

        # DROPPING COLUMNS WITH ONLY ONE UNIQUE VALUE
        num_unique = self.data.nunique()
        one_variable = num_unique[num_unique == 1]
        not_one_variable = num_unique[num_unique > 1].index
        self.data = self.data[not_one_variable]
        logging.info(f"Dropped columns with one unique value: {list(one_variable.index)}")

        # PCA
        if self.config["pca"]:
            logging.info("Starting PCA...")
            features = self.data.drop('Label', axis = 1)
            attacks = self.data['Label']

            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)

            size = len(features.columns) // 2
            ipca = IncrementalPCA(n_components = size, batch_size = 500)
            for batch in np.array_split(scaled_features, len(features) // 500):
                ipca.partial_fit(batch)

            logging.info(f'information retained: {sum(ipca.explained_variance_ratio_):.2%}')
            transformed_features = ipca.transform(scaled_features)
            self.data = pd.DataFrame(transformed_features, columns = [f'PC{i+1}' for i in range(size)])
            self.data['Label'] = attacks.values
            logging.info("PCA complete.")

        logging.info("...preprocessing DONE")