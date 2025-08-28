import os
import requests
import logging
import zipfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.config import DATA_DIR, CIC_IDS2017_URL, EXTENDED_CIC_IDS2017_URL, ANALYSIS_PATH, CICIDS2017_DEFAULT_CONFIG, CIC_IDS2017_CLASSES_MAPPING
from src.data._base import TabularDataset
from src.data.utils import concat_and_save_csv, features_correction, category_column_ascii_correction

class CICIDS2017(TabularDataset):
    """
    Class for downloading, loading, and processing the CIC-IDS2017 dataset.
    """

    # extended_dataset_folder = os.path.join("TrafficLabelling_", "TrafficLabelling ")
    dataset_folder = "MachineLearningCVE"
    processed_data_folder = "processed"

    def __init__(self, **kwargs):
        """
        Args:
            extended (bool): If True, load the extended dataset (TrafficLabelling_);
                             otherwise, load the normal dataset (MachineLearningCVE).
        """

        # Config
        self.default_config = CICIDS2017_DEFAULT_CONFIG.copy()
        self.default_config.update(kwargs)

        # Routes
        self.raw_data_dir = os.path.abspath(os.path.join(DATA_DIR, "CIC-IDS2017", self.dataset_folder))
        self.processed_data_dir = os.path.abspath(os.path.join(DATA_DIR, "CIC-IDS2017", self.processed_data_folder))
        url = CIC_IDS2017_URL

        # Super
        name = self._get_id()
        super().__init__(self.processed_data_dir, f"{name}.csv", url)
        self.extended = False

    def _get_id(self) -> str:

        sorted_values = sorted(self.default_config.items())
        return "cic_ids2017_" + '_'.join([f"{key}({value})" for key, value in sorted_values])

    def _download(self) -> None:
        """
        Downloads and extracts dataset files.
        """
        if not self.url:
            logging.error("No URL provided for dataset download.")
            return

        os.makedirs(self.raw_data_dir, exist_ok=True)
        zip_path = os.path.join(self.raw_data_dir, os.path.basename(self.url))
        
        logging.info(f"Downloading dataset from {self.url}...")
        response = requests.get(self.url, stream=True)
        if response.status_code == 200:
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    filename = os.path.basename(member)
                    if filename:
                        with zip_ref.open(member) as source:
                            target_path = os.path.join(self.raw_data_dir, filename)
                            with open(target_path, "wb") as target:
                                target.write(source.read())
            os.remove(zip_path)
        else:
            logging.error(f"Failed to download dataset from {self.url}.")

    def _collect(self) -> None:
        """
        Retrieves and merges the raw dataset if a saved version does not exist.
        """

        if not os.path.exists(self.processed_data_dir):
            self._download()
    
        if not os.path.isfile(self.file_path):
            logging.info(f"{self.file_path} not found. Creating dataset from individual CSV files in {self.raw_data_dir}.")
            self.data = concat_and_save_csv(self.raw_data_dir, self.file_name, encoding="latin-1", save_in_path=self.processed_data_dir, sep=",", return_df=True)
        else:
            self.data = pd.read_csv(self.file_path)

    def _preprocess(self) -> None:
        """
        Applies dataset-specific preprocessing, such as feature correction and cleaning.
        """

        logging.info("Starting preprocessing...")

        # Correct feature names
        features_correction(self.data)
        logging.info("Feature names corrected.")

        # Correct categories names
        category_column_ascii_correction(self.data, "Label")
        logging.info("Labels names corrected.")

        # Drop duplicates
        self.data.drop_duplicates(inplace=True)
        logging.info("Duplicates dropped.")

        # Infinite values
        ## Replacing with NaN is a good answer ?? (Should check the feature, infinity might have a meaning)
        # maybe dropping might be better ??
        self.data.replace([np.inf, -np.inf], np.nan, inplace = True)

        # Missing values
        ## Dropping might be better ?? the rows with infinites have them always both in Flow Bytes and Fow Packages
        ## also, only around 6% missing values (including infinites)
        ## Filling with median might affect performance or learning ??
        med_flow_bytes = self.data['Flow_Bytes/s'].median()
        med_flow_packets = self.data['Flow_Packets/s'].median()
        self.data['Flow_Bytes/s'].fillna(med_flow_bytes, inplace = True)
        self.data['Flow_Packets/s'].fillna(med_flow_packets, inplace = True)

        if self.data.isna().sum().sum() > 0:
            logging.warning("Dropping rows with NaN values after cleaning.")
        self.data.dropna(inplace=True)
        logging.info("Missing values handled.")

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
        if self.default_config["pca"]:
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

        if self.default_config["classes_mapping"]:

            self.data["Label"] = self.data['Label'].map(CIC_IDS2017_CLASSES_MAPPING)
            logging.info("Classes mapping complete.")

        logging.info("...preprocessing DONE")

    def analysis(self) -> None:
        """
        Method for executing an analysis over the CIC-IDS20217 dataset and saving results on memory
        """

        save_directory = os.path.join(ANALYSIS_PATH, self._get_id())
        os.makedirs(save_directory, exist_ok=True)

        logging.info(f"Starting CIC-IDS2017 {'(extended)' if self.extended else ''} analysis...")

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
        benign_count = attack_counts.get('BENIGN', 0)
        attack_count = attack_counts.drop('BENIGN').sum()
        plt.figure(figsize=(6, 6))
        plt.pie(
            [benign_count, attack_count],
            labels=['BENIGN', 'Other Attacks'],
            autopct='%1.1f%%',
            colors=['lightblue', 'lightcoral'],
            textprops={'fontsize': 12}
        )
        plt.title('BENIGN vs Other attacks distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, "attacks_distribution_benign_vs_others.png"))

        attack_only_counts = attack_counts.drop('BENIGN')
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

        # BOXPLOT OF FEATURES FOR EACH ATTACK / FIRMS FOR EACH ATTACK
        logging.info("Distribution of features...")
        os.makedirs(os.path.join(save_directory, "attacks_boxplots"), exist_ok=True)
        for attack_type in self.data['Label'].unique():

            attack_data = self.data[self.data['Label'] == attack_type]
            plt.figure(figsize=(20, 20))
            sns.boxplot(data = attack_data.drop(columns = ['Label']), orient = 'h')
            plt.title(f'Boxplot of Features for Attack Type: {attack_type}')
            plt.xlabel('Feature Value')
            plt.savefig(os.path.join(save_directory, "attacks_boxplots", f"{str(attack_type).strip().replace(' ', '_')}.png"))
        
        logging.info(f"...analysis DONE")

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    dataset = CICIDS2017()
    dataset.load()
