import logging 
import mlflow
import pandas as pd
from src.model_dev import(
    RandomForestModel 
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin
from zenml import step 
from zenml.client import Client
from .config import ModelNameConfig 
from typing import Any 
import joblib

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.Series,
    X_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    config : ModelNameConfig,
)-> ClassifierMixin :
    """
    Args:
    x_train: pd.DatFrame
    x_test: pd.DataFrame
    y_train: pd.Series'
    y_train: pd.Series
    Returns:
        model: ClassifierMixin
    """
    try:
        model = None
        
        if config.model_name == "random_forest":
            mlflow.sklearn.autolog()
            model = RandomForestModel()
            trained_model = model.train(X_train, y_train)  # Train the model
            return trained_model
        else:
            raise ValueError("Model name not supported")
    except Exception as e:
        logging.error(e)
        raise e  
