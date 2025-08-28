import os


DATA_DIR = "data"
ANALYSIS_PATH = "results"

#KAGGLE
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", None)
KAGGLE_KEY = os.environ.get("KAGGLE_KEY", None)

# CIC-IDS2017
EXTENDED_CIC_IDS2017_URL = "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/GeneratedLabelledFlows.zip"
CIC_IDS2017_URL = "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip"

CIC_IDS2017_CLASSES_MAPPING = {
    'BENIGN': 'BENIGN',
    'DDoS': 'DDoS',
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'PortScan': 'Port Scan',
    'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    'Bot': 'Bot',
    'Web Attack  Brute Force': 'Web Attack',
    'Web Attack  Sql Injection': 'Web Attack',
    'Web Attack  XSS': 'Web Attack',
    'Infiltration': 'Infiltration',
    'Heartbleed': 'Heartbleed'
}

CICIDS2017_DEFAULT_CONFIG = {
    "pca": True,
    "classes_mapping": True
}

# N-BaIoT
N_BAIOT_URL = 'https://www.kaggle.com/api/v1/datasets/download/mkashifn/nbaiot-dataset'
N_BAIOT_KAGGLE_ENDPOINT = 'mkashifn/nbaiot-dataset'
N_BAIOT_TARGET_COLUMN_NAME = "Label"

N_BAIOT_DEFAULT_CONFIG = {
    "64_to_32_quantization": True,
    "pca": True,
    "classes_mapping": True
}

N_BAIOT_CLASSES_MAPPING = {
    'benign': "benign",
    'gafgyt-combo': "gafgyt",
    'gafgyt-junk': "gafgyt",
    'gafgyt-scan': "gafgyt",
    'gafgyt-tcp': "gafgyt",
    'gafgyt-udp': "gafgyt",
    'mirai-ack': "mirai",
    'mirai-scan': "mirai",
    'mirai-syn': "mirai",
    'mirai-udp': "mirai",
    'mirai-udpplain': "mirai"
}

# CIC-UNSW
CICUNSW_URL = "http://cicresearch.ca/CICDataset/CIC-UNSW/Dataset/CICFlowMeter_out.csv"

CICUSNW_DEFAULT_CONFIG = {
    "pca": True,
}

# DATA AUGMENTATION
SMOTE_CONFIG = {
    "class_samples_threshold": 0.3,
    "n": 10,
    "alpha":  0.7,
    "beta": 0.3
}

TABPFN_DATA_GENERATOR_CONFIG = {
    "class_samples_threshold": 0.3,
    "t": 1,
    "n_permutations": 3
}
TABPFN_DATA_GENERATOR_MODEL_CONFIG = {
    "n_estimators": 1,
    "average_before_softmax": False,
    "device": "cuda",
    "ignore_pretraining_limits": True,
    "inference_precision": "auto",
    "fit_mode": "fit_preprocessors",
    "memory_saving_mode": True,
    "random_state": 0,
    "n_jobs": -1,
}
TABPFN_DATA_GENERATOR_EXPERT_MODEL_CONFIG = { # IT IS RECOMMENDED TO NOT CHANGE THIS PARAMETERS
    "CLASS_SHIFT_METHOD": "shuffle",
    "FEATURE_SHIFT_METHOD": "shuffle",
    "FINGERPRINT_FEATURE": True,
    "MAX_NUMBER_OF_CLASSES": 10,
    "MAX_NUMBER_OF_FEATURES": 500,
    "MAX_NUMBER_OF_SAMPLES": 10000,
    "MAX_UNIQUE_FOR_CATEGORICAL_FEATURES": 30,
    "MIN_UNIQUE_FOR_NUMERICAL_FEATURES": 4,
    "OUTLIER_REMOVAL_STD": "auto",
    "POLYNOMIAL_FEATURES": 'no',
    "SUBSAMPLE_SAMPLES": None,
    "USE_SKLEARN_16_DECIMAL_PRECISION": False
}