import logging 
import mlflow 
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np 
import pandas as pd
from src.evaluation import Recall,F1Score,Accuracy,Precision
from sklearn.base import ClassifierMixin
from typing_extensions import Annotated
from src.model_dev import RandomForestModel
from zenml import step 
from zenml.client import Client 
experiment_tracker = Client().active_stack.experiment_tracker
from typing import Tuple, Any
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

@step(experiment_tracker=experiment_tracker.name)
def evaluation(
    model: ClassifierMixin, X_test :pd.Series,y_test:pd.Series
) -> Tuple[ Annotated[float,"accuracy"],Annotated[float,"f1_score"],Annotated[float,"precision"],Annotated[float,"recall"]]:
    
      
    try:
        cv = joblib.load('vectorizer.joblib')
        X_test = cv.transform(X_test)
        print('this is evaluation',X_test)
        prediction = model.predict(X_test)
        
        f1_score_class = F1Score()   
        f1_score = f1_score_class.calculate_score(y_test,prediction)
        print("The f1 score is ",f1_score)
        mlflow.log_metric("f1_score",f1_score)
        
        accuracy_class = Accuracy()
        accuracy = accuracy_class.calculate_score(y_test,prediction)
        mlflow.log_metric("accuracy",accuracy)
        
        precision_class = Precision()
        precision = precision_class.calculate_score(y_test,prediction)
        mlflow.log_metric("precision",precision)
        
        recall_class = Recall()
        recall = recall_class.calculate_score(y_test,prediction)
        mlflow.log_metric("recall",recall)
        
        model_path = "model.joblib"
        joblib.dump(model, model_path)

        # Log the serialized model as an artifact
        mlflow.log_artifact(model_path, "model")
        
        return recall,accuracy,f1_score,precision
    except Exception as e:
        logging.error(e)
        raise e 
    
    