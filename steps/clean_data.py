import logging 
from typing import Tuple 
import pandas as pd 
from src.data_cleaning import(
    DataCleaning,
    DataDivideStrategy,
    DataPreprocessingStrategy
)
from typing_extensions import Annotated 
from zenml import step 

@step
def clean_data(
    data: pd.DataFrame,
) -> Tuple[
    Annotated[pd.Series,"X_train"],
    Annotated[pd.Series,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"]
]:
    """
    Data cleaning class which preprocess the data and divides it into train and test data
    """
    try:
        preprocess_strategy = DataPreprocessingStrategy()
           
        data_cleaning = DataCleaning(data,preprocess_strategy)
        preprocess_strategy = data_cleaning.handle_data()
        print(preprocess_strategy)
        
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocess_strategy,divide_strategy)
        X_train,X_test,y_train,y_test = data_cleaning.handle_data()
        return X_train,X_test,y_train,y_test
    except Exception as e:
        logging.error(e)
        raise e 
    