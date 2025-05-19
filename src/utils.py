import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_feature_importance(importance_df, top_n=10):
    """
    Plot feature importance scores
    
    Parameters:
    -----------
    importance_df : pandas.DataFrame
        DataFrame containing feature importance scores
    top_n : int
        Number of top features to plot
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df.head(top_n),
        x='importance',
        y='feature'
    )
    plt.title(f'Top {top_n} Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    return plt

def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    labels : list
        List of class labels
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    return plt

def save_results(results, filepath):
    """
    Save model results to a CSV file
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results
    filepath : str
        Path to save the results
    """
    results_df = pd.DataFrame(results)
    results_df.to_csv(filepath, index=False)
    return results_df 