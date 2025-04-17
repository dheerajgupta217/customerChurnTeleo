# Databricks notebook source
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("ChurnPrediction").getOrCreate()

# Load dataset
data_path = "/test/FD_testSrc_testDS.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Save as Delta table
df.write.format("delta").mode("overwrite").save("/mnt/datalake/delta/telco_churn")


# COMMAND ----------

from pyspark.sql.functions import col, when

# Load Delta table
df = spark.read.format("delta").load("/mnt/datalake/delta/telco_churn")

# Handle missing values
df = df.na.fill({"TotalCharges": 0})

# Convert target variable to binary
df = df.withColumn("Churn", when(col("Churn") == "Yes", 1).otherwise(0))

# Save processed data
df.write.format("delta").mode("overwrite").save("/mnt/datalake/delta/processed_telco_churn")

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType

# Load processed data
df = spark.read.format("delta").load("/mnt/datalake/delta/processed_telco_churn")
# Fix casting issue
df = df.withColumn("TotalCharges", col("TotalCharges").cast(DoubleType()))

# Optional: Handle any rows where conversion failed and created nulls
df = df.na.fill({"TotalCharges": 0.0})
# List of categorical columns
categorical_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
                    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                    "PaperlessBilling", "PaymentMethod"]

# Index categorical columns
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index") for col in categorical_cols]

# Assemble features
assembler = VectorAssembler(inputCols=[col+"_index" for col in categorical_cols] + ["tenure", "MonthlyCharges", "TotalCharges"],
                            outputCol="features")

# Apply transformations
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=indexers + [assembler])
model = pipeline.fit(df)
df_transformed = model.transform(df)

# Save features
df_transformed.select("features", "Churn").write.format("delta").mode("overwrite").save("/mnt/datalake/delta/features_telco_churn")


# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
df_transformed.show()


# COMMAND ----------

dbutils.library.restartPython()

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


# COMMAND ----------

from pyspark.sql.functions import rand, when
# Load model
# model_uri = "runs:/<run_id>/logistic_regression_model"
model_uri = 'runs:/4c29409e0fb94b369e5ed1bc67b02282/logistic_regression_model'
lr_model = mlflow.spark.load_model(model_uri)

# Load test data
test_df = spark.read.format("delta").load("/mnt/datalake/delta/features_telco_churn")
.filter("split = 'test'")

# Make predictions
predictions = lr_model.transform(test_df)

# Evaluate
auc = evaluator.evaluate(predictions)
print(f"AUC: {auc}")


# COMMAND ----------

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
