import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pipeline.deployment_pipeline import prediction_service_loader
from run_deployment import main

from materializer.custom_materializer import cs_materializer
from zenml.integrations.mlflow.services import MLFowDeploymentService
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer

MODEL_NAME = 'model'
PIPELINE_NAME = "continuous_deployment_pipeline"
PIPELINE_STEP_NAME = "mlflow_model_deployer_step"

@st.cache
def load_model():
    # Load the trained model using ZenML
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    model_service = model_deployer.find_model_server(
        pipeline_name=PIPELINE_NAME,
        pipeline_step_name=PIPELINE_STEP_NAME,
        model_name=MODEL_NAME,
        running=True,
    )[0]

    return model_service

def main():
    st.title("Financial Sentiment Analysis Pipeline using Zenml ")
    user_input = st.text_area("Enter a sentence: ")
    if st.button("Predict Sentiment"):
        model_service = prediction_service_loader()
        prediction = predict_sentiment(model_service,user_input)
        st.write("Predicted Sentiment: ",prediction)
        
def predict_sentiment(model_service, user_input):
    input_data = {"sentence ":user_input}
    prediction = model_service.predict(json.dumps(input_data))
    return prediction 

if __name__ == "__main__":
    main()