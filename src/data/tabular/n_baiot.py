import os
import logging
import glob
import re

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
import seaborn as sns
import matplotlib.pyplot as plt

from src.data.config import DATA_DIR, KAGGLE_USERNAME, KAGGLE_KEY, N_BAIOT_URL, N_BAIOT_KAGGLE_ENDPOINT, N_BAIOT_DEFAULT_CONFIG, N_BAIOT_CLASSES_MAPPING, N_BAIOT_TARGET_COLUMN_NAME, ANALYSIS_PATH
from src.data._base import TabularDataset

class Kaggle_Dataset(TabularDataset):

    def __init__(self, data_dir, file_name, url = None):
        """
        """
        
        from kaggle.api.kaggle_api_extended import KaggleApi

        self.kaggle_api = KaggleApi

        super().__init__(data_dir, file_name, url)

class N_BaIoT(Kaggle_Dataset):
    """
    Class for downloading, loading, and processing the N-BaIoT dataset using Kaggle.
    """

    dataset_folder = "N-BaIoT"

    def __init__(self, **kwargs) -> None:

        # Config
        self.config = N_BAIOT_DEFAULT_CONFIG.copy()
        self.config.update(kwargs)

        self.raw_data_dir = os.path.abspath(os.path.join(DATA_DIR, self.dataset_folder, "raw"))
        self.processed_data_dir = os.path.abspath(os.path.join(DATA_DIR, self.dataset_folder, "processed"))
        url = N_BAIOT_URL
        
        super().__init__(self.processed_data_dir, self._get_id() + ".csv", url)

    def _get_id(self) -> str:
        
        sorted_values = sorted(self.config.items())
        return "n_baiot_" + '_'.join([f"{key}({value})" for key, value in sorted_values])
    
    def _download(self) -> None:
        """
        Dowloads and extracts data files
        """

        if not KAGGLE_USERNAME or not KAGGLE_KEY:
            raise Exception("KAGGLE_USERNAME or KAGGLE_KEY environment variables not set")
        
        if not self.url:
            logging.error("No URL provided for dataset download.")
            return
        
        os.makedirs(self.raw_data_dir, exist_ok=True)

        api = self.kaggle_api()
        api.authenticate()

        logging.info(f"Downloading dataset from {self.url}")
        try:
            api.dataset_download_files(N_BAIOT_KAGGLE_ENDPOINT, path=self.raw_data_dir, unzip=True)
            logging.info("Dowload successful")
        except Exception as e:
            logging.info(f"Failed to download dataset, error: {e}")

        os.makedirs(self.processed_data_dir, exist_ok=True)

    def _collect(self) -> None:
        """
        Retrieves the data whether it is by dowloading it or by cleaning the raw files.
        """
        
        if not os.path.exists(self.processed_data_dir):
            self._download()
    
        if not os.path.isfile(self.file_path):
            logging.info(f"{self.file_path} not found. Collecting and cleaning raw data in {self.raw_data_dir}...")
            self.data = self._clean(target_column_name=N_BAIOT_TARGET_COLUMN_NAME)
            self.data.to_csv(self.file_path, sep=",", index=False)
            logging.info(f"Saving cleaned data...DONE")
        else:
            logging.info(f"Collecting data from saved ({self.file_path})")
            self.data = pd.read_csv(self.file_path)

        logging.info("Done collecting data")

    def _clean(self, target_column_name: str = N_BAIOT_TARGET_COLUMN_NAME) -> pd.DataFrame:
        """
        Method for cleaning the data.
        """
        
        raw_files = [f for f in glob.glob(os.path.join(self.raw_data_dir, "*.csv"), recursive=False) if re.match(r"[1-9]*\.", os.path.basename(f))]
        
        dfs = []
        for path in raw_files:
            target_class = os.path.basename(path)[2:-4].replace(".", "-")
            df = pd.read_csv(path)
            df[target_column_name] = target_class
            dfs.append(df)
            logging.info(f"Cleaning {os.path.basename(path)}...DONE")

        combined_df = pd.concat(dfs, ignore_index=True)

        return combined_df
    
    def _preprocess(self) -> None:
        
        logging.info("Starting preprocessing...")

        self.data.drop_duplicates(inplace=True)
        logging.info("Dropping duplicates...DONE")


        if self.config["64_to_32_quantization"]:
            logging.info("Dowcasting float64 and int64 to float32 and int32...")
            for col in self.data.columns:
                col_type = self.data[col].dtype
                if col_type != object:
                    c_min = self.data[col].min()
                    c_max = self.data[col].max()
                    if str(col_type).find('float') >= 0 and c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        try:
                            self.data[col] = self.data[col].astype(np.float32)
                        except ValueError:
                            logging.warning(f"Failed to downcast {col} to float32")

                    elif str(col_type).find('int') >= 0 and c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        self.data[col] = self.data[col].astype(np.int32)
            logging.info("Quantization...DONE")

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

        if self.config["classes_mapping"]:

            self.data["Label"] = self.data['Label'].map(N_BAIOT_CLASSES_MAPPING)
            logging.info("Classes mapping complete.")

        logging.info("...preprocessing DONE")

    def analysis(self) -> None:
        """
        Method for executing an analysis over the N-BaIoT dataset and saving results on memory
        """

        save_directory = os.path.join(ANALYSIS_PATH, self._get_id())
        os.makedirs(save_directory, exist_ok=True)

        logging.info(f"Starting N-BaIoT analysis...")

        # CORRELATION MATRIX
        logging.info("Correlation matrix...")
        corr = self.data.corr(numeric_only = True).round(2)
        corr.style.background_gradient(cmap = 'coolwarm', axis = None).format(precision = 2)
        fig, ax = plt.subplots(figsize = (24, 24))
        sns.heatmap(corr, cmap = 'coolwarm', annot = False, linewidth = 0.5)
        plt.title('Correlation Matrix', fontsize = 18)
        plt.savefig(os.path.join(save_directory, "correlation_matrix.png"))

        # DISTRIBUTION OF ATTACKS
        attack_counts = self.data['Label'].value_counts()
        benign_count = attack_counts.get('benign', 0)
        attack_count = attack_counts.drop('benign').sum()
        plt.figure(figsize=(6, 6))
        plt.pie(
            [benign_count, attack_count],
            labels=['benign', 'Other Attacks'],
            autopct='%1.1f%%',
            colors=['lightblue', 'lightcoral'],
            textprops={'fontsize': 12}
        )
        plt.title('benign vs Other attacks distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, "attacks_distribution_benign_vs_others.png"))

        attack_only_counts = attack_counts.drop('benign')
        total_attacks = attack_only_counts.sum()
        percentages = (attack_only_counts / total_attacks) * 100
        labels = [
            label if percentages[i] >= 1 else '' 
            for i, label in enumerate(attack_only_counts.index)
        ]
        def autopct_func(pct):
            return ('%1.1f%%' % pct) if pct >= 1 else ''
        pastel_colors = sns.color_palette('pastel', len(attack_only_counts))
        plt.figure(figsize=(10, 10))
        plt.pie(
            attack_only_counts.values,
            labels=labels,
            autopct=autopct_func,
            colors=pastel_colors,
            textprops={'fontsize': 8}
        )
        plt.title('Distribution of attack types')
        plt.legend(attack_only_counts.index, loc='best', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, "attacks_distribution_attacks_only.png"))

        
