# Databricks notebook source
# Register model
mlflow.register_model(model_uri, "ChurnPredictionModel")

# Transition model to production
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.transition_model_version_stage(
    name="ChurnPredictionModel",
    version=1,
    stage="Production"
)
