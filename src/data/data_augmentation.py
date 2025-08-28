import logging
import numpy as np
from pprint import pformat
from collections import Counter
from typing import Optional

import torch
from sklearn.preprocessing import LabelEncoder
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.config import ModelInterfaceConfig
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

from src.data.config import SMOTE_CONFIG, TABPFN_DATA_GENERATOR_MODEL_CONFIG, TABPFN_DATA_GENERATOR_EXPERT_MODEL_CONFIG, TABPFN_DATA_GENERATOR_CONFIG

class TabPFNDataGenerator:
    """
    Synthetic data generator based on TabPFN model
    """

    generator: TabPFNUnsupervisedModel

    def __init__(self):
        
        self.tabpfn_clf = TabPFNClassifier(
            **TABPFN_DATA_GENERATOR_MODEL_CONFIG,
            inference_config = ModelInterfaceConfig(
                **TABPFN_DATA_GENERATOR_EXPERT_MODEL_CONFIG
            )
        )
        self.tabpfn_regr = TabPFNRegressor(
            **TABPFN_DATA_GENERATOR_MODEL_CONFIG,
            inference_config = ModelInterfaceConfig(
                **TABPFN_DATA_GENERATOR_EXPERT_MODEL_CONFIG
            )
        )

        self.fitted = False
        self.label_encoder = LabelEncoder()

    def generate(self, n_samples: int, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None, threshold: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates `n_samples` of data based on the provided data to be fitted.

        Following the configuration of the generator, data is synthetically produced. Not all classes are generated, only those with a number of samples relative to the `threshold` 
        value and the amount of samples of the predominant class. The classes of the samples generated are random.

        Once fitted it can be used to generate more samples without needing more examples.

        Args:
            n_smaples: Number of samples to be generated
            x: Features to fit the generator with
            y: Classes to fit the generator with
            threshold: Factor that multiplies the max amount of samples in a class to determine the classes being augmented.
        """

        if not self.fitted:

            assert x is not None and y is not None

            unique_labels, counts = np.unique(y, return_counts=True)
            max_n_samples = max(counts)

            if not threshold:
                threshold = TABPFN_DATA_GENERATOR_CONFIG["class_samples_threshold"]
            assert isinstance(threshold, float)
            assert 0 < threshold <= 1

            threshold_value = int(max_n_samples * threshold)
            augmentation_classes = [label for label, count in dict(zip(unique_labels, counts)).items() if count <= threshold_value]

            mask = np.isin(y, augmentation_classes)
            base_x = x[mask]
            base_y = y[mask]
            base_y = self.label_encoder.fit_transform(base_y)
            base_combined = np.column_stack((base_x, base_y.reshape(-1, 1)))

            n_classes = len(augmentation_classes)
            assert n_classes <= TABPFN_DATA_GENERATOR_EXPERT_MODEL_CONFIG["MAX_NUMBER_OF_CLASSES"], "Number of classes to be augmented exceeded, try with a lower threshold value"

            self.generator = TabPFNUnsupervisedModel(
                tabpfn_clf=self.tabpfn_clf,
                tabpfn_reg=self.tabpfn_regr
            )

            logging.info("Fitting generator...")
            logging.info(f"Classes to be fitted: {augmentation_classes}")
            self.generator.fit(
                torch.from_numpy(base_combined),

            )

            self.fitted = True

        logging.info(f"Generating {n_samples} samples...")
        generated_samples: torch.Tensor = self.generator.generate_synthetic_data(
            n_samples, t=TABPFN_DATA_GENERATOR_CONFIG["t"],
            n_permutations=TABPFN_DATA_GENERATOR_CONFIG["n_permutations"]
        ).numpy()

        generated_x = generated_samples[:, :-1]
        generated_y = generated_samples[:, -1]
        
        valid_classes = np.arange(len(self.label_encoder.classes_))
        mask = np.isin(generated_y, valid_classes)
        generated_x = generated_x[mask]
        generated_y = generated_y[mask]
        generated_y = self.label_encoder.inverse_transform(generated_y.astype(int))

        new_x = np.concatenate([x, generated_x], axis=0)
        new_y = np.concatenate([y, generated_y], axis=0)

        unique_classes, counts = np.unique(new_y, return_counts=True)
        class_counts = {cls: int(count) for cls, count in zip(unique_classes, counts)}
        logging.info("Final counts after Data Augmentation:\n%s", pformat(class_counts))

        return new_x, new_y

def smote(x: np.ndarray, y: np.ndarray, threshold: Optional[float] = None, n: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Specifically for DL model, initiall conceived for TabNET. Performs data augmentation for clasification using a modified version of SMOTE.

    Given the data divided into features and labels, the data augmentation will be performed to those classes that have a certain percentange of samples 
    less than the class with the higher number of samples. The augmentation will be performed `augmentations` times, increasing the amount of data for the class each time.

    Args:
        x: The features of the data.
        y: The labels of the data associated to the features.
        threshold: The percentage that indicates whether to augment a class samples or not depending of the class with the most number of samples. (0, 1].
        n: The amount of times to augment the data.

    Returns:
        The tuple feeatures-labels of the original data plus the augmented samples.
    """

    logging.info("Performing SMOTE for data augmentation...")

    if not threshold:
        threshold = SMOTE_CONFIG["class_samples_threshold"]
    
    assert isinstance(threshold, float)
    assert 0 < threshold <= 1

    if not n:
        n = SMOTE_CONFIG["n"]

    assert isinstance(n, int)

    # Obtain labels and their counts as well as the max amount of samples in a class
    unique_labels, counts = np.unique(y, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    max_n_samples = max(counts)

    # Get the threshold value classes for augmentation
    threshold_value = int(max_n_samples * threshold)
    augmentation_classes = [label for label, count in label_counts.items() if count < threshold_value]

    # Perform augmentation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    augmented_x = torch.from_numpy(x.copy()).to(device)
    augmented_y = y.copy()

    logging.info("Starting Data Augmentation...")
    for _ in range(n):

        new_x, new_y = _classification_SMOTE_step(
            torch.from_numpy(x.copy()).to(device),
            y.copy(),
            SMOTE_CONFIG["alpha"],
            SMOTE_CONFIG["beta"],
            augmentation_classes,
            seed=np.random.randint(1, 200)
        )

        augmented_x = torch.cat((augmented_x, new_x), dim=0)
        augmented_y = np.concatenate((augmented_y, new_y), axis=0)

        # Update counts and classes being augmented
        new_unique_labels, new_counts = np.unique(y, return_counts=True)
        new_label_counts = dict(zip(new_unique_labels, new_counts))
        label_counts = dict(Counter(label_counts) + Counter(new_label_counts))
        augmentation_classes = [label for label, count in label_counts.items() if count < threshold_value]

    unique_classes, counts = np.unique(augmented_y, return_counts=True)
    class_counts = {cls: int(count) for cls, count in zip(unique_classes, counts)}
    logging.info("Final counts after Data Augmentation:\n%s", pformat(class_counts))

    return augmented_x.cpu().numpy(), augmented_y

def _classification_SMOTE_step(
    x: torch.torch.Tensor,
    y: np.ndarray,
    alpha: float,
    beta: float,
    augmentation_classes: list[str],
    seed: int) -> tuple[torch.Tensor, np.ndarray]:
    """
    Personalized version of SMOTE based on the pytorch-tabnet library implementation ClassificationSMOTE.

    Specifically designed to augment all the samples based on specific classes and not a percentage of the data, at the same time that the augmented samples can utilize any of the samples 
    from classes that are not being augmented.

    Args:
        x: Features samples
        y: Labels associated to the features samples
        alpha: alpha for the beta distribution
        beta: beta for the beta distribution
        augmentation_classes: List of classes being augmented
        seed: Seed for random numbers

    Returns:
        The tuples features-labels of augmented data (not including the original samples).
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    batch_size = x.shape[0]
    device = x.device
    mask_to_change = np.isin(y, augmentation_classes)
    mask_to_change_t = torch.tensor(mask_to_change, device=device)

    np_betas = np.random.beta(alpha, beta, batch_size) / 2 + 0.5
    random_betas = torch.from_numpy(np_betas).to(device).float()
    index_permute = torch.randperm(batch_size, device=device)

    x[mask_to_change_t] = random_betas[mask_to_change_t, None] * x[mask_to_change_t]
    x[mask_to_change_t] += (1 - random_betas[mask_to_change_t, None]) * x[index_permute][mask_to_change_t].view(x[mask_to_change_t].size())  # noqa

    return x[mask_to_change_t], y[mask_to_change]


