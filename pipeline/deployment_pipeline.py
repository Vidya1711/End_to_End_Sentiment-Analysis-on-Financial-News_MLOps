import json 
import os 
import numpy as np 
import joblib
from joblib import load
from scipy.sparse import csr_matrix
import pandas as pd
from materializer.custom_materializer import cs_materializer
from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_df
from steps.model_train import train_model
from zenml import pipeline,step 
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer
)
from sklearn.feature_extraction.text import CountVectorizer
from zenml.integrations.mlflow.services import MLFlowDeploymentService 
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from steps.ingest_data import ingest_df
from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.model_train import train_model
from .utils import get_data_for_test

docker_settings = DockerSettings(required_integrations=[MLFLOW])


class DeploymentTriggerConfig(BaseParameters):
    min_accuracy:float = 0.0
    
@step(enable_cache=False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data 

  
    
@step(enable_cache=False)
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
):
    return accuracy >= config.min_accuracy

class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    pipeline_name : str
    step_name : str 
    config : DeploymentTriggerConfig
    running : bool = True

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    # get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    print(existing_services)
    return existing_services[0]

@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    if service is None:
        raise ValueError("MLFlowDeploymentService is None. Check your deployment setup.")
    
    # Start the service
    service.start(timeout=10)
    try: 
        # Load data and preprocess
        data = json.loads(data)
        data.pop("columns" )
        data.pop("index" )
        print('this is data',data)
        columns_for_df = ["Headlines"]
        df = pd.DataFrame(data["data"], columns = columns_for_df)
        df = df['Headlines'].apply(lambda x: ''.join(x))
        print('this is pred df',df)
        print('this is pred type',type(df))
        vectorizer = joblib.load('vectorizer.joblib')
        transformed_data = vectorizer.transform(df)
        transformed_data = pd.DataFrame(transformed_data.getnnz(axis=1))
        print(df)
        print('This is type of predictor',type(transformed_data))
        # Make predictions 
        prediction = service.predict(transformed_data)
        
        return prediction
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    finally:
        # Make sure to stop the service even if an exception occurs
        service.stop()
    


@pipeline(enable_cache=False,settings={'docker':docker_settings})
def continous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0,
    workers: int=1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, X_test, y_train, y_test)
    accuracy,precision,f1_score,recall = evaluation(model,X_test,y_test)
    deployment_decision = deployment_trigger(accuracy)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )
    

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts together
    data = dynamic_importer()
    service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    prediction = predictor(service=service, data=data)
    return prediction



# import numpy as np
# import pandas as pd
# import json
# import joblib
 
# from zenml import pipeline, step
# from zenml.config import DockerSettings
# from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
# from zenml.integrations.constants import MLFLOW
# from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import ( MLFlowModelDeployer)

# from zenml.integrations.mlflow.services import MLFlowDeploymentService
# from zenml.integrations.mlflow.steps import mlflow_model_deployer_step 
# from zenml.steps import BaseParameters, Output

# from .utils import get_data_for_test
# from steps.ingest_data import ingest_df
# from steps.clean_data import clean_data
# from steps.model_train import train_model
# from steps.evaluation import evaluation

# docker_settings = DockerSettings(required_integrations=[MLFLOW])

# class DeploymentTriggerConfig(BaseParameters):
#     """Deployment Trigger Config"""
#     min_accuracy: float = 0.0

# @step(enable_cache=False)
# def dynamic_importer() -> str:
#     data = get_data_for_test()
#     return data 

# @step
# def deployment_trigger(accuracy: float,config: DeploymentTriggerConfig):
#     """
#     Implements a simple model deployment trigger that looks at the
#     input model accuracy and decides if it is good enough to deploy
#     """
#     return accuracy >= config.min_accuracy

# class MLFlowDeploymentLoaderStepParameters(BaseParameters):
#     """MLflow deployment getter parameters

#     Attributes:
#         pipeline_name: name of the pipeline that deployed the MLflow prediction
#             server
#         step_name: the name of the step that deployed the MLflow prediction
#             server
#         running: when this flag is set, the step only returns a running service
#         model_name: the name of the model that is deployed
#     """

#     pipeline_name: str
#     step_name: str
#     running: bool = True

# @step(enable_cache=False)
# def prediction_service_loader(
#     pipeline_name: str,
#     pipeline_step_name: str,
#     running: bool = True,
#     model_name: str = "model"
# ) -> MLFlowDeploymentService:
#     """Get the prediction service started by the deployment pipeline.

#     Args:
#         pipeline_name: name of the pipeline that deployed the MLflow prediction
#             server
#         step_name: the name of the step that deployed the MLflow prediction
#             server
#         running: when this flag is set, the step only returns a running service
#         model_name: the name of the model that is deployed
#     """
#     # get the MLflow model deployer stack component
#     mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
#     # fetch existing services with same pipeline name, step name and model name

#     existing_services = mlflow_model_deployer_component.find_model_server(
#         pipeline_name = pipeline_name,
#         pipeline_step_name = pipeline_step_name,
#         model_name = model_name,
#         running = running
#     )

#     if not existing_services:
#         raise RuntimeError(
#             f"No MLflow prediction service deployed by the "
#             f"{pipeline_step_name} step in the {pipeline_name} "
#             f"pipeline for the '{model_name}' model is currently "
#             f"running."
#         )
#     return existing_services[0]

# @step
# def predictor(
#     service: MLFlowDeploymentService,
#     data: str,
#     tfidf_vectorizer: StepArtifact  # Updated to StepArtifact
# ) -> np.ndarray:
#     """Run an inference request against a prediction service"""

#     service.start(timeout=10)
#     data = json.loads(data)
#     data.pop("columns")
#     data.pop("index")
#     columns_for_df = ["Sentiments", "Headlines"]
#     df = pd.DataFrame(data["data"], columns=columns_for_df)

#     # Load the TF-IDF vectorizer from the artifact
#     loaded_tfidf_vectorizer = tfidf_vectorizer.get_artifact()

#     # Apply TF-IDF vectorizer to the new data
#     new_data_tfidf = loaded_tfidf_vectorizer.transform(df['Headlines'])

#     # Convert the preprocessed data to the required format (if needed)
#     json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
#     data = np.array(json_list)

#     # Make predictions using the deployed model
#     prediction = service.predict(data)
#     return prediction



# @pipeline(enable_cache=False, settings={"docker": docker_settings})
# def continuous_deployment_pipeline( 
#     data_path: str,
#     min_accuracy: float = 0.0,
#     workers: int = 1,
#     timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
# ):
#     df = ingest_df(data_path=data_path)
#     X_train, X_test, y_train, y_test = clean_data(df)
#     model = train_model(X_train, X_test, y_train, y_test)
#     accuracy,precision,f1_score,recall = evaluation(model,X_test,y_test)
#     deployment_decision = deployment_trigger(accuracy)
#     mlflow_model_deployer_step(model=model,
#                                deploy_decision=deployment_decision,
#                                workers=workers,
#                                timeout=timeout)
    
# @pipeline(enable_cache=False, settings={"docker": docker_settings})
# def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
#     # Load the pre-trained TF-IDF vectorizer
#     tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    
#     # Link all the steps artifacts together
#     batch_data = dynamic_importer()
#     model_deployment_service = prediction_service_loader(
#         pipeline_name=pipeline_name,
#         pipeline_step_name=pipeline_step_name,
#         running=False,
#     )
#     predictor(
#         service=model_deployment_service,
#         data=batch_data,
#         tfidf_vectorizer=tfidf_vectorizer
#     )