# Databricks notebook source
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "default")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

print(f"Hola desde el bundle -> {catalog}.{schema}")
display(spark.range(5))
