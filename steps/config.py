from zenml.steps import BaseParameters 

class ModelNameConfig(BaseParameters):
    model_name: str = 'random_forest'
    fine_tuning: bool = False