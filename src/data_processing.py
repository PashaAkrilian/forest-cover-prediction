import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load the forest cover type dataset
    
    Parameters:
    -----------
    file_path : str
        Path to the dataset file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded dataset
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocess the data for model training
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test
    """
    # Separate features and target
    X = df.drop('Cover_Type', axis=1)
    y = df['Cover_Type']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test 