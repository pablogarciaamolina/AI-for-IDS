import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_accuracies(
    scores: list[float],
    names: list[str]
) -> plt.Figure:
    """
    Function for plotting a report of accuracies

    Args:
        scores: List containing the scores of the models
        names: Names of the models from which the accuracies where obtained

    Returns:
        A figure containing the report
    """
    
    assert len(scores) == len(names), f"{len(scores)} scores and {len(names)} provided"
    
    palette = sns.color_palette('Blues', n_colors = 2)
    fig, ax = plt.subplots(figsize = (9, 3))
    ax.barh(names, scores, color = palette)
    ax.set_xlim([0, 1])
    ax.set_xlabel('Accuracy Score')
    ax.set_title('Models Comparison')

    for i, v in enumerate(scores):
        ax.text(v + 0.01, i, str(round(v, 3)), ha = 'left', va = 'center')

    return fig
