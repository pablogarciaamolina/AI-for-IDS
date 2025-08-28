# AI for Intrusion Detection Systems

A framework for research on AI-based IDS.

Code base for the paper **"Foundation Models for Cybersecurity: A Comprehensive Multi-Modal Evaluation of TabPFN and TabICL for Tabular Intrusion Detection"**.


[![Python](https://img.shields.io/badge/python-3.10+-blue)]()

## Table of Contents

1. [Project Structure](#project-structure)  
2. [How To Use](#how-to-use)  
3. [Modules - Specs](#modules---specs)  
   - [DATA](#data)  
     - [Modules Hierarchy](#modules-hierarchy)  
     - [Configurations](#configurations)  
       - [Kaggle Configuration](#kaggle-configuration)  
       - [CIC-IDS2017 Configuration](#cic-ids2017-configuration)  
       - [N-BaIoT Configuration](#n-baiot)  
       - [Data Augmentation](#data-augmentation)  
   - [MODELS](#models)  
     - [Modules Hierarchy](#modules-hierarchy-1)  
     - [Configurations](#configurations-1)  
       - [TabNet Configuration Parameters](#tabnet-configuration-parameters)  
       - [ML Models Configuration Parameters](#ml-models-configuration-parameters)  
       - [TabPFN Configuration Parameters](#tabpfn-configuration-parameters)  
       - [LLMs Configuration Parameters](#llms-configuration-parameters)  
   - [PIPELINES](#pipelines)  
     - [Modules Hierarchy](#modules-hierarchy-2)

## Project Structure

```plaintext
TFG/
├── src/
│   ├── data/
│   │   ├── tabular/
│   │   │   ├── ...
│   │   │   ├── cic_ids2017.py
│   │   │   ├── cic_unsw.py
│   │   │   ├── n_baiot.py
│   │   ├── ...
│   │   ├── config.py
│   │   ├── data_augmentation.py
│   │   ├── utils.py
│   ├── models/
│   │   ├── API/
│   │   │   ├── ...
│   │   │   ├── llm.py
│   │   ├── trainable/
│   │   │   ├── ...
│   │   │   ├── ml.py
│   │   │   ├── tabicl.py
│   │   │   ├── tabnet.py
│   │   │   ├── tabpfn.py
│   │   ├── ...
│   │   ├── config.py
│   │   ├── utils.py
│   ├── pipelines/
│   │   ├── ...
│   │   ├── pipelines.py
│   │   ├── utils.py
├── .gitgnore
├── README.md
└── .*.ipynb (example notebooks)
```

## How To Use

This framework is based on the idea of modues for dealing with all the steps on the development of AI models. Just by selecting one compatible module from each category, one can establish the model, train on a dataset and measure the performance.

The code is divided into three main categories: ``data``, ``models`` and ``pipelines``. Each contains a series of modules that can be used and easily changed for research. Also, this structure allows for great expandability, making very convenient the extension of functionalities.

The repository includes code notebooks with examples on how to use each of the modules in a end-to-end scenario.

## Modules - Specs

The specification of how to use each module are contained in the code. For configuration, ``data`` and ``models`` contain their own files that allows the user to change the behaviour of their modules. The pipelines can be changed using the arguments of each method.

### DATA

This category wraps all the modules related to the datasets. Each dataset counts with its own module, that allows for easy download, preprocessing and adaptation.

#### Modules hierarchy
```
BaseDataset
├── TabularDataset
│   ├── CIC-IDS2017
│   └── N-BaIoT
│   └── CIC-UNSW
└── (other possible types)
```

#### Configurations

##### Kaggle configuration
- ``KAGGLE_USERNAME``: Kaggle username
- ``KAGGLE_KEY``:  Kaggle key
    - ⚠️ Note that, even if these are represented as constants, Kaggle authentification REQUIERES THAT THEY ARE SET AS ENVIRONMENT VARIABLES.
    - ⚠️ Note that Kaggle allows fo rother ways of authentification. However, this method of environment variables is the most convenient. When trying to load a dataset that is imported from kaggle (e.g. N-BaIoT), an error will be raised if the authentification is not properly set.

##### CIC-IDS2017 configuration

- ``CIC_IDS2017_CLASSES_MAPPING``: Mapping of classes for CIC-IDS2017.

- ``CICIDS2017_DEFAULT_CONFIG``
    - **pca**: Whether to use PCA by default.
    - **classes_mapping**: Whether to use class mapping by default.


##### N-BaIoT

- ``N_BAIOT_DEFAULT_CONFIG``
    - **64_to_32_quantization**: Whether to use 64bit-to-32 bit quantization by default.
    - **pca**: Whether to use PCA by default.
    - **classes_mapping**: Whether to use class mapping by default.

- ``N_BAIOT_CLASSES_MAPPING``: Mapping of classes for N-BaIoT.

##### CIC-UNSW configuration

- ``CICUNSW_DEFAULT_CONFIG``
    - **pca**: Whether to use PCA by default.

##### DATA AUGMENTATION

- ``SMOTE_CONFIG``
    - **class_samples_threshold**: Percentage that indicates whether to augment a class samples or not depending of the class with the most number of samples.
    - **n**: The amount of times to augment the data.
    - **alpha**:  alpha for the beta distribution.
    - **beta**: beta for the beta distribution.

- ``TABPFN_DATA_GENERATOR_CONFIG``
    - **class_samples_threshold**: Percentage that multiplies the max amount of samples in a class to determine the classes being augmented.
    - **t**: Tenperature parameter for sampling. Hieher values produce more diverse samples, lower values produce more deterministic samples.
    - **n_permutations**: Number of feature permutations to use for generation More permutations may provide more robust results but increase computation time.

- ``TABPFN_DATA_GENERATOR_MODEL_CONFIG`` See [TabPFN](#tabpfn-configuration-parameters)

- ``TABPFN_DATA_GENERATOR_EXPERT_MODEL_CONFIG`` See [TabPFN](#tabpfn-configuration-parameters)

### MODELS

The models are divided into LLMs (API based models) and trainable models, like ML or DL models.

For further explanation the behaviour or construction of this models, look into the notebook examples or the specific docstrings.

#### Modules hierarchy
```
BaseModel
├── TrainableModel
│   ├── SklearnTrainableModel
│   │   ├── MLClassifier
│   │   │   ├── PreConfigured_DecisionTree
│   │   │   ├── PreConfigured_KNeighbors
│   │   │   ├── PreConfigured_LinearSVC
│   │   │   ├── PreConfigured_LogisticRegression
│   │   │   └── PreConfigured_RandomForest
│   │   ├── TabNetModel
│   │   ├── TabPFNModel
│   │   └── TabICLModel
│   └── (other kinds of models, like torch-based DL models)
└── LLModel
    ├── Mistral
    └── Gemini
```

#### Configurations

Some models can have their configuration or their training modified via arguments. However, all of them can be modified in full detail using the configuration file.

Below one can find the configuration parameters for every model explained.

##### TabNet Configuration Parameters


- ``TABNET_PRETRAINER_CONFIG``
    - **n_d**: Dimension of the prediction layer
    - **n_a**: Dimension of the attention layer
    - **n_steps**: Number of successive steps or layers in the network
    - **gamma**: Scaling factor for attention updates (usually between 1.0 and 2.0)
    - **n_independent**: Number of independent GLU layer in each GLU block
    - **n_shared**: Number of independent GLU layer in each GLU block
    - **lambda_sparse**: Sparsity loss parameter
    - **optimizer_fn**: Optimizer
    - **optimizer_params**: Parameters for the optimizer

- ``TABNET_PRETRAINING_PARAMS``
    - **max_epochs**: Maximum pretraining epochs
    - **batch_size**: Batch size. 
        - ⚠️ Note that if the batch size is higher than the amount of data samples provided (both training or validation) an error will be raised.
    - **patience**: Number of epochs with no improvement after which training will be stopped

- ``TABNET_CONFIG``
    - **n_d**: Dimension of the prediction layer
    - **n_a**: Dimension of the attention layer
    - **n_steps**: Number of successive steps or layers in the network,
    - **gamma**: Scaling factor for attention updates (usually between 1.0 and 2.0)
    - **n_independent**: Number of independent GLU layer in each GLU block
    - **n_shared**: Number of independent GLU layer in each GLU block
    - **lambda_sparse**: Sparsity loss parameter
    - **optimizer_fn**: Optimizer
    - **optimizer_params**: Parameters for the optimizer
    - **scheduler_fn**: Scheduler
    - **scheduler_params**: Scheduler parameters
    - **mask_type**: masking function

- ``TABNET_TRAINING_PARAMS``
    - **eval_metric**: Evaluation metric
    - **max_epochs**: Maximum number of training epochs
    - **batch_size**: Batch size.
        - ⚠️ Note that if the batch size is higher than the amount of data samples provided (both training or validation) an error will be raised.
    - **patience**: Number of epochs with no improvement after which training will be stopped

##### ML Models Configuration Parameters

- ``LOGISTIC_REGRESSION_CONFIG``
    - **multi_class**: {'auto', 'ovr', 'multinomial'}
    - **max_iter**: Maximum number of iterations taken for the solvers to converge
    - **solver**: Algorithm to use in the optimization problem
    - **C**: Inverse of regularization strength; must be a positive float
    - **random_state**: Used when solver` == 'sag', 'saga' or 'liblinear' to shuffle the data

- ``LINEAR_SVC_CONFIG``
    - **C**: Regularization parameter. The strength of the regularization is inversely proportional to C.
    - **random_state**: Controls the pseudo random number generation for shuffling the data for the dual coordinate descent.
    - **tol**: Tolerance for stopping criteria
    - **max_iter**: Maximum number of iterations to be run.

- ``RANDOM_FOREST_CONFIG``
    - **n_estimators**: The number of trees in the forest.
    - **max_depth**: The maximum depth of the tree
    - **max_features**: The number of features to consider when looking for the best split
    - **random_state**: Controls both the randomness of the bootstrapping of the samples used when building trees and the sampling of the features to consider when looking for the best split at each node

- ``KNEIGHBORS_CONFIG``
    - **n_neighbors**: Number of neighbors to use

- ``DECISION_TREE_CONFIG``
    - **max_depth**: The maximum depth of the tree

##### TabPFN Configuration Parameters

- ``TABPFN_CONFIG``
    - **n_estimators**: The number of estimators in the TabPFN ensemble
    - **balance_probabilities**: Whether to balance the probabilities based on the class distribution in the training data
    - **average_before_softmax**:  Whether to average the predictions of the estimators before applying the softmax function
    - **device**: The device to use for inference with TabPFN
    - **ignore_pretraining_limits**: Whether to ignore the pre-training limits of the model
    - **inference_precision**: The precision to use for inference
    - **fit_mode**: Determines how the data is preprocessed and cached for inference
    - **memory_saving_mode**: Enable GPU/CPU memory saving mode
    - **random_state**: Controls the randomness of the model
    - **n_jobs**: The number of workers for tasks that can be parallelized across CPU cores

- ``TABPFN_EXPERT_CONFIG``
    - **CLASS_SHIFT_METHOD**: The method used to shift classes during preprocessing for ensembling to emulate the effect of invariance to class order
    - **FEATURE_SHIFT_METHOD**: The method used to shift features during preprocessing for ensembling to emulate the effect of invariance to feature position
    - **FINGERPRINT_FEATURE**: Whether to add a fingerprint feature to the data
    - **MAX_NUMBER_OF_CLASSES**: The number of classes seen during pretraining for classification.
    - **MAX_NUMBER_OF_FEATURES**: The number of features that the pretraining was intended for
    - **MAX_NUMBER_OF_SAMPLES**: The number of samples that the pretraining was intended for
    - **MAX_UNIQUE_FOR_CATEGORICAL_FEATURES**: The maximum number of unique values for a feature to be considered categorical
    - **MIN_UNIQUE_FOR_NUMERICAL_FEATURES**: The minimum number of unique values for a feature to be considered numerical
    - **OUTLIER_REMOVAL_STD**: The number of standard deviations from the mean to consider a sample an outlier
    - **POLYNOMIAL_FEATURES**: The number of 2 factor polynomial features to generate and add to the original data before passing the data to TabPFN
    - **SUBSAMPLE_SAMPLES**: Subsample the input data sample/row-wise before performing any preprocessing and the TabPFN forward pass
    - **USE_SKLEARN_16_DECIMAL_PRECISION**: Whether to round the probabilities to float 16 to match the precision of scikit-learn

- ``TABPFN_PARAMS``
    - **predicting_batch_size**: Batch size for predicting using batches. Use -1 for single batch.

- ``TABPFN_MANY_CLASS_CONFIG``
    - **alphabet_size**: Maximum number of classes the base estimator can handle. Has to be lower or equal than MAX_NUMBER_OF_CLASSES
    - **n_estimators**: Number of base estimators to train. None for automatic detection, int for  specific number
    - **n_estimators_redundancy**: Redundancy factor for the auto-calculated number of estimators,
    - **random_state**: Controls randomization used to initialize the codebook.

##### TabICL Configuration Parameters

- ``TABICL_CONFIG``
    - **n_estimators**: Number of ensemble members
    - **norm_methods**: Normalization methods to try
    - **feat_shuffle_method**: Feature permutation strategy
    - **class_shift**: Whether to apply cyclic shifts to class labels
    - **outlier_threshold**: Z-score threshold for outlier detection and clipping
    - **softmax_temperature**:Ccontrols prediction confidence
    - **average_logits**: Whether ensemble averaging is done on logits or probabilities
    - **use_hierarchical**: Enable hierarchical classification for datasets with many classe
    - **batch_size**: Process this many ensemble members together (reduce RAM usage)
    - **use_amp**: Use automatic mixed precision for faster inference
    - **model_path**: Where the model checkpoint is stored
    - **allow_auto_download**: whether automatic download to the specified path is allowed
    - **checkpoint_version**: The version of pretrained checkpoint to use
    - **device**: Specify device for inference
    - **random_state**: Random seed for reproducibility
    - **n_jobs**: Number of threads to use for PyTorch
    - **verbose**: Print detailed information during inference

- ``TABICL_PARAMS``
    - **predicting_batch_size**: Batch size for predicting using batches. Use -1 for single batch.

##### LLMs Configuration Parameters
- ``MISTRAL_API_KEY``: Mistral API key

- ``GEMINI_API_KEY``: Gemini API key

- ``BACKOFF_MAX_TRIES``: The maximum number of attempts to make before giving up

- ``BACKOFF_FACTOR``: The amount by which the backoff time increases on each successive backoff.

- ``GEMINI_GENERATION_CONFIG``
    - **top_p**: Tokens are selected from the most to least probable until the sum of their probabilities equals this value.
    - **temperature**: Value that controls the degree of randomness in token selection. Lower temperatures are good for prompts that require a less open-ended or creative response, while higher temperatures can lead to more creative results.
    - **candidate_count**: Number of response variations to return.

- ``MISTRAL_GENERATION_CONFIG``
    - **top_p**: Nucleus sampling, where the model considers the results of the tokens with `top_p` probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or `temperature` but not both.
    - **temperature**: What sampling temperature to use, we recommend between 0.0 and 0.7. Higher values like 0.7 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    - **n**: Number of completions to return for each request, input tokens are only billed once.

### PIPELINES

#### Modules hierarchy
```
BasePipeline
├── TTPipeline
└── TAPipeline
```