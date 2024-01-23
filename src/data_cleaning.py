import logging 
from abc import ABC, abstractclassmethod 
from typing import Union 
import nltk
nltk.download('punkt')
import joblib
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import * 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import string,time
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """
    
    @abstractclassmethod
    def handle_data(self,data:pd.DataFrame) :
        pass 
    
class DataPreprocessingStrategy(DataStrategy):
    """
    Data preprocessing strategy which preprocesses the data
    """
    
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # Rename columns
            columns = data.columns.tolist()
            columns[1] = 'Headlines'
            columns[0] = 'Sentiments'
            data.columns = columns
            
            # Convert headlines to lowercase
            data['Headlines'] = data['Headlines'].apply(lambda x: x.lower())
            
            # Remove stopwords
            sw_list = stopwords.words('english')
            data['Headlines'] = data['Headlines'].apply(lambda x: ' '.join([item for item in x.split() if item not in sw_list]))
            
            # Remove HTML tags
            def remove_tags(raw_text):
                cleaned_text = re.sub(re.compile('<.*?>'), '', raw_text)
                return cleaned_text
            data['Headlines'] = data['Headlines'].apply(remove_tags)
            
            # Remove punctuation
            exclude = string.punctuation
            def remove_punc1(text):
                return text.translate(str.maketrans('', '', exclude))
            data['Headlines'] = data['Headlines'].apply(remove_punc1)
            
            stemmer = PorterStemmer()

            def stem_headline(headline):
                words = word_tokenize(headline)
                stemmed_words = [stemmer.stem(word) for word in words]
                return " ".join(stemmed_words)

            # Apply stemming to the entire 'Headlines' column
            data['Headlines'] = data['Headlines'].apply(stem_headline)
            # Label encoding for sentiments
            le = LabelEncoder()
            data['Sentiments'] = le.fit_transform(data['Sentiments'])
            return data
        
        except Exception as e:
            logging.error(e)
            raise e
           
class DataDivideStrategy(DataStrategy):
    """
    Divides the data into train and test
    """
    def handle_data(self, data: pd.DataFrame) ->  pd.Series:
        try:
            X = data['Headlines']
            y = data['Sentiments']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e 
            
class DataCleaning:
    """
    Data Cleaning class which preprocess the data and divides it into train and test data
    """
    def __init__(self,data:pd.DataFrame,strategy:DataStrategy)-> None:
        """Initialize the DataCleaning class with a specific strategy"""
        self.df = data 
        self.strategy = strategy 
        
    def handle_data(self) -> Union[pd.DataFrame,pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_data(self.df)
       
    