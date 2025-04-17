# Databricks notebook source
from pyspark.ml.feature import StringIndexer, VectorAssembler

# Load processed data
df = spark.read.format("delta").load("/mnt/datalake/delta/processed_telco_churn")

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
