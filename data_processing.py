# Databricks notebook source
from pyspark.sql.functions import col, when

# Load Delta table
df = spark.read.format("delta").load("/mnt/datalake/delta/telco_churn")

# Handle missing values
df = df.na.fill({"TotalCharges": 0})

# Convert target variable to binary
df = df.withColumn("Churn", when(col("Churn") == "Yes", 1).otherwise(0))

# Save processed data
df.write.format("delta").mode("overwrite").save("/mnt/datalake/delta/processed_telco_churn")
