# Databricks notebook source
# MAGIC %pip install mlflow==2.7.1 typing_extensions==4.5.0
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
print("MLflow version:", mlflow.__version__)


# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.spark

# Load features
df = spark.read.format("delta").load("/mnt/datalake/delta/features_telco_churn")

# Split data
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Initialize model
lr = LogisticRegression(featuresCol="features", labelCol="Churn")

# Start MLflow run
with mlflow.start_run():
    # Train model
    lr_model = lr.fit(train_df)
    
    # Log model
    mlflow.spark.log_model(lr_model, "logistic_regression_model")
    
    # Evaluate model
    predictions = lr_model.transform(test_df)
    evaluator = BinaryClassificationEvaluator(labelCol="Churn")
    auc = evaluator.evaluate(predictions)
    
    # Log metric
    mlflow.log_metric("AUC", auc)
