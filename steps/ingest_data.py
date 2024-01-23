import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Ingesting the data from the data_path
    """
    def __init__(self, data_path: str, encoding: str = 'ISO-8859-1'):
        """
        Args:
            data_path: path to the data
        """
        self.data_path = data_path
        self.encoding = encoding  # Store encoding as an instance variable

    def get_data(self):
        """
        Ingesting the data from the data_path
        """
        logging.info(f"Ingesting data from {self.data_path}")
        df = pd.read_csv(self.data_path, encoding=self.encoding)  # Use the stored encoding
        return df

@step
def ingest_df(data_path: str, encoding: str = 'ISO-8859-1') -> pd.DataFrame:
    """
    Ingesting the data from the data_path

    Args:
        data_path: path to the data
    Returns:
        dict: Dictionary with the ingested data
    """
    try:
        ingest_df = IngestData(data_path, encoding=encoding)
        df = ingest_df.get_data()
        return df 
    except Exception as e:
        logging.error(e)
        raise e
