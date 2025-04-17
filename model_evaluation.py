# Databricks notebook source
# Load model
model_uri = "runs:/<run_id>/logistic_regression_model"
lr_model = mlflow.spark.load_model(model_uri)

# Load test data
test_df = spark.read.format("delta").load("/mnt/datalake/delta/features_telco_churn").filter("split = 'test'")

# Make predictions
predictions = lr_model.transform(test_df)

# Evaluate
auc = evaluator.evaluate(predictions)
print(f"AUC: {auc}")
