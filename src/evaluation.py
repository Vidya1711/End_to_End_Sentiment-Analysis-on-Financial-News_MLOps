import logging
from abc import ABC,abstractclassmethod

import numpy as np
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score

class Evaluation(ABC):
    """
    Abstract Class defining the strategy for evaluating model performance
    """
    @abstractclassmethod
    def calculate_score(self,y_true:np.ndarray,y_pred:np.ndarray) ->float:
        pass 
    

        
class Accuracy(Evaluation):
    def calculate_score(self,y_true:np.ndarray,y_pred:np.ndarray)-> float:
        try:
            logging.info("Entered the accuracy_score method of the Accuracy class")
            accuracy = accuracy_score(y_true,y_pred)
            logging.info("The accuracy score value is:" + str(accuracy))
            print("accuracy_score",accuracy)
            return accuracy
        
        except Exception as e:
            logging.error(f"Error in accuracy evaluation: {e}")
            raise e 
        
class Recall(Evaluation):
    def calculate_score(self,y_true: np.ndarray,y_pred:np.ndarray)-> float:
        try:
            recallscore = recall_score(y_true,y_pred,average='micro')
            print("recall score: ",recallscore)
            return recallscore 
        except Exception as e:
            logging.error(f"Error in recall evaluation: {e}")
            raise e 
        
class Precision(Evaluation):
    def calculate_score(self,y_true: np.ndarray,y_pred:np.ndarray)-> float:
        try:
            precisionscore = precision_score(y_true,y_pred,average='micro')
            print("precision",precisionscore)
            return precisionscore 
             
        except Exception as e:
            logging.error(f"Error in recall evaluation: {e}")
            raise e 
        
        
class F1Score(Evaluation):
    def calculate_score(self,y_true:np.ndarray,y_pred:np.ndarray)-> float:
        try:
            f1score =  f1_score(y_true,y_pred,average='micro')
            print("f1_score: ",f1score)
            return f1score
        except Exception as e:
            logging.error(f"Erroe in f1_score: {e}")
            raise e 
        
        
