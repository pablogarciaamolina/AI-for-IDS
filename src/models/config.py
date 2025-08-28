import os
import torch

ML_SAVING_PATH = os.path.join("models", "ml")
TABNET_SAVING_PATH = os.path.join("models", "tabnet")
TABPFN_SAVING_PATH = os.path.join("models", "tabpfn")
TABICL_SAVING_PATH = os.path.join("models", "tabicl")

# TABNET
TABNET_PRETRAINER_CONFIG = {
    "n_d": 10,
    "n_a": 10,
    "n_steps": 5,
    "gamma": 1.3,
    "n_independent": 4,
    "n_shared": 4,
    "lambda_sparse": 1e-3,
    "optimizer_fn": torch.optim.AdamW,
    "optimizer_params": {"lr": 2e-2},
    "mask_type": "entmax",
}
TABNET_PRETRAINING_PARAMS = {
    "max_epochs": 100,
    "batch_size": 1024,
    "patience": 10,
}

TABNET_CONFIG = {
    "n_d": 10,
    "n_a": 10,
    "n_steps": 5,
    "gamma": 1.3,
    "n_independent": 4,
    "n_shared": 4,
    "lambda_sparse": 1e-3,
    "optimizer_fn": torch.optim.AdamW,
    "optimizer_params": {"lr": 2e-2, "weight_decay": 1e-5},
    "scheduler_fn": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 50, "gamma": 0.9},
    "mask_type": "entmax",
}
TABNET_TRAINING_PARAMS = {
    "eval_metric": ["accuracy"],
    "max_epochs": 200,
    "batch_size": 1024,
    "patience": 20,
}

# ML MODELS
LOGISTIC_REGRESSION_CONFIG = {
    "multi_class": "multinomial",
    "max_iter": 15000,
    "solver": 'sag',
    "C": 100,
    "random_state": 0
}

LINEAR_SVC_CONFIG = {
    "C": 0.8,
    "random_state": 0,
    "tol": 1e-2,
    "max_iter": 5000,
}

RANDOM_FOREST_CONFIG = {
    "n_estimators": 15, 
    "max_depth": 8, 
    "max_features": 20, 
    "random_state": 0
}

KNEIGHBORS_CONFIG = {
    "n_neighbors": 8
}

DECISION_TREE_CONFIG = {
    "max_depth": 8
}

# TABPFN
TABPFN_CONFIG = {
    "n_estimators": 4,
    "balance_probabilities": False,
    "average_before_softmax": False,
    "device": "auto",
    "ignore_pretraining_limits": False,
    "inference_precision": "auto",
    "fit_mode": "fit_preprocessors",
    "memory_saving_mode": True,
    "random_state": 0,
    "n_jobs": -1,
}
TABPFN_EXPERT_CONFIG = { # IT IS RECOMMENDED TO NOT CHANGE THIS PARAMETERS
    "CLASS_SHIFT_METHOD": "shuffle",
    "FEATURE_SHIFT_METHOD": "shuffle",
    "FINGERPRINT_FEATURE": True,
    "MAX_NUMBER_OF_CLASSES": 10,
    "MAX_NUMBER_OF_FEATURES": 500,
    "MAX_NUMBER_OF_SAMPLES": 100000,
    "MAX_UNIQUE_FOR_CATEGORICAL_FEATURES": 30,
    "MIN_UNIQUE_FOR_NUMERICAL_FEATURES": 4,
    "OUTLIER_REMOVAL_STD": "auto",
    "POLYNOMIAL_FEATURES": 'no',
    "SUBSAMPLE_SAMPLES": None,
    "USE_SKLEARN_16_DECIMAL_PRECISION": False
}
TABPFN_MANY_CLASS_CONFIG = {
    "alphabet_size": 5, # Has to be <= than MAX_NUMBER_OF_CLASSES
    "n_estimators": None, # None for automatic detection, int for  specific number
    "n_estimators_redundancy": 4,
    "random_state": 0
}
TABPFN_PARAMS = {
    "predicting_batch_size": 20000 # -1 for one batch (no batch predicting)
}

# TABICL
TABICL_CONFIG = {
    "n_estimators": 16,
    "norm_methods": ["none", "power"],
    "feat_shuffle_method": "latin",
    "class_shift": True,
    "outlier_threshold": 4.0,
    "softmax_temperature": 0.9,
    "average_logits": True,
    "use_hierarchical": True,
    "batch_size": 8,
    "use_amp": True,
    "model_path": None,
    "allow_auto_download": True,
    "checkpoint_version": "tabicl-classifier-v1.1-0506.ckpt",
    "device": None,
    "random_state": 42,
    "n_jobs": None,
    "verbose": True,
    "inference_config": None,
}

TABICL_PARAMS = {
    "predicting_batch_size": 50000
}

# LLMs
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", None)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)

BACKOFF_MAX_TRIES = 15
BACKOFF_FACTOR = 2

GEMINI_GENERATION_CONFIG = {
    "top_p": 0.8,
    "temperature": 0.9,
    "candidate_count": 1
}

MISTRAL_GENERATION_CONFIG = {
    "top_p": 0.8,
    "temperature": 0.9,
    "n": 1
}