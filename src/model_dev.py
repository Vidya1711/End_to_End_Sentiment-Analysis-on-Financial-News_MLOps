from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from abc import ABC, abstractmethod 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import ClassifierMixin
import logging
import pandas as pd 
import joblib
 
 
class Model(ABC):
    """
    This abstract base class define a common interface for model development strategies.
    
    Attributes:
        None 
        
    Methods:
        develop_model: It develops a machine learning model using the provided training data
    """
    
    @abstractmethod
    def train(self,X_train,y_train):
        """
        Trains the model on the given data
        
        Args:
            x_train: Training data
            y_train: Target data
        """
        pass 
    
     
    
    
class RandomForestModel(Model):
    """
    XGBoostModel that implements the Model interface.
    """
    
    def train(self, X_train, y_train, **kwargs):
        try:
            cv = CountVectorizer(max_features=3000)
            print('this is model dev',X_train)
            print('this is model dev',type(X_train))
            X_train = cv.fit_transform(X_train).toarray()
            # X_test = cv.transform(X_test).toarray()
            # X_test = pd.DataFrame(X_test)
            # X_train = pd.DataFrame(X_train)
            print('this is model dev',X_train)
            print('this is model dev',type(X_train))
            model =  RandomForestClassifier()
            model.fit(X_train, y_train)
            logging.info("Model training completed")
            joblib.dump(cv, 'vectorizer.joblib')
            return model 

        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e 

             


        