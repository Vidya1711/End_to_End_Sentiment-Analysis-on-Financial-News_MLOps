from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluation

@pipeline(enable_cache=False)
def train_pipeline(data_path: str):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, X_test, y_train, y_test)
    accuracy,precision,f1_score,recall = evaluation(model,X_test,y_test)