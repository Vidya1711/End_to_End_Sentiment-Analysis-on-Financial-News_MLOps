import streamlit as st
import pandas as pd 
import joblib
import json 
import numpy as np
from pipeline.deployment_pipeline import prediction_service_loader



def prediction():
    st.title('Sentiment Analysis Model')
    
    try:
        headline = st.text_input('Headline', min_value=0)

        df = {
            'Headline': headline
        }
    
        
        data = pd.Series(headline)
        if st.button('Predict'):
            service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=False,
        )
        df = pd.DataFrame(
            {
            'Headline':[headline]
            }
        )
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)
        st.success(f"User Sentiment is  :{pred}") 
            
    except Exception as e:
        st.error(e)
        
        


if __name__ == '__main__':
    prediction()