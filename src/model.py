import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

class ForestCoverPredictor:
    def __init__(self):
        self.model = None
        self.best_params = None
        
    def train(self, X_train, y_train):
        """
        Train the XGBoost model with hyperparameter tuning
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training labels
        """
        # Define parameter grid for tuning
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Initialize XGBoost classifier
        xgb_model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=7,
            random_state=42
        )
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Save best parameters and model
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        return self.best_params
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Parameters:
        -----------
        X : numpy.ndarray
            Features to predict
            
        Returns:
        --------
        numpy.ndarray
            Predicted labels
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            Test features
        y_test : numpy.ndarray
            True labels
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'classification_report': report
        }
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance scores
        
        Parameters:
        -----------
        feature_names : list
            List of feature names
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing feature importance scores
        """
        importance_scores = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        })
        return importance_df.sort_values('importance', ascending=False) 