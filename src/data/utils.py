import os
import glob
import numpy as np
import re

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def features_correction(data: pd.DataFrame) -> None:
    """
    Inplace function for correcting DataFrame features names.

    Args:
        data: DataFrame
    """

    new_col_names = {col: col.strip().replace(" ", "_") for col in data.columns}
    data.rename(columns=new_col_names, inplace=True)

def category_column_ascii_correction(data: pd.DataFrame, col_name: str) -> None:
    """
    Inplace function for correcting the categories in a column of a DataFrame, encoding them to ascii.

    Args:
        data: DataFrame
    """

    data[col_name] = data[col_name].apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', str(x)))


def concat_and_save_csv(path: str, name: str, encoding: str = "utf-8", save_in_path: bool|str = False, sep: str = ",", return_df: bool = False) -> pd.DataFrame | None:
    """
    Reads all the CSV files in `path` and saves their concatenation as a CSV file named `name`

    Args:
        path: Path for the directory in wich the CSV files are stored
        name: Name under which to save the CSV file resulted of the concatenation of the other files

        encoding: Encoding for reading the CSV files. Also used for saving. Defaults to `utf-8`
        save_in_path: Where to save the file. If True it will be saved in the same directory as the rest of CSVs, else it will be saved in the root directory under `name`. If a path is provided it will be saved in it. Defaults to False
        sep: Separator used for reading and saving.
        return_df: Whether to return the merged dataframe. Defaults to False

    Return:
        Either a DataFrame of the data or None
    """

    files = glob.glob(os.path.join(path, "*.csv"), recursive=False)
    df = pd.concat((pd.read_csv(file, encoding=encoding, sep=sep) for file in files))

    if isinstance(save_in_path, bool):
        df.to_csv(os.path.join(path, name) if save_in_path else name, sep=sep, index=False)
    else:
        os.makedirs(save_in_path, exist_ok=True)
        df.to_csv(os.path.join(save_in_path, name), sep=sep, index=False)

    if return_df:
        return df
    else:
        del df

def sample(data: pd.DataFrame, p: float = 0.2) -> pd.DataFrame:
    """
    Samples (without replacement) a percentage of the data in a pandas DataFrame

    Args:
        data: DataFrame to sample
        p: Percentage to sample. Must be between 0% and 100%
    """

    assert (0 <= p) and (p <= 1)

    sample_size = int(p * len(data))
    sampled_data = data.sample(n = sample_size, replace = False, random_state = 0).reset_index(drop=True)

    return sampled_data

def balanced_sample(df: pd.DataFrame, category_col: str, n_per_class: int) -> pd.DataFrame:
    """
    Draws a balanced sample from a DataFrame based on a categorical variable.
    
    Args:
        df: DataFrame
        category_col: Column name of the categorical variable
        n_per_class: Number of samples per category
    Returns:
        A balanced DataFrame sample
    """
    return df.groupby(category_col).apply(lambda x: x.sample(n=min(len(x), n_per_class))).reset_index(drop=True)

def jsonize_rows(data: pd.DataFrame) -> list[str]:
    """
    Transforms input datatframe into json format row by row

    Args:
        data: Dataframe containing the data
    """

    jsonized_rows = [row.to_json() for _, row in data.iterrows()]

    return jsonized_rows

def encode_labels(data: pd.DataFrame, label_col: str) -> tuple[np.ndarray]:
    """
    Encodes categorical labels with a label encoder

    Args:
        data: Pandas Dataframe containing the data
        label_col: Name of the column containing the categorical labels

    Returns:
        Data separated into features and encoded labels
    """

    x = data.drop(columns=[label_col]).values
    y = data[label_col].values
    y = LabelEncoder().fit_transform(y)

    return x, y