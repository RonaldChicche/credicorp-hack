# Databricks notebook source
# MAGIC %md
# MAGIC # Exploración de datos
# MAGIC Recorre catálogos, schemas y tablas, y perfila una tabla a elegir.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("table", "")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
table = dbutils.widgets.get("table")

# COMMAND ----------

# MAGIC %md ## 1. Qué hay disponible

# COMMAND ----------

display(spark.sql("SHOW CATALOGS"))

# COMMAND ----------

display(spark.sql(f"SHOW SCHEMAS IN {catalog}"))

# COMMAND ----------

display(spark.sql(f"SHOW TABLES IN {catalog}.{schema}"))

# COMMAND ----------

# MAGIC %md ## 2. Ubicaciones externas (S3)

# COMMAND ----------

display(spark.sql("SHOW EXTERNAL LOCATIONS"))

# COMMAND ----------

# MAGIC %md ## 3. Perfilado de una tabla
# MAGIC Indica `table` en el widget de arriba.

# COMMAND ----------

if not table:
    print("Define el widget 'table' para perfilar una tabla.")
else:
    fqn = f"{catalog}.{schema}.{table}"
    df = spark.table(fqn)
    print(f"{fqn}: {df.count()} filas, {len(df.columns)} columnas")
    df.printSchema()
    display(df.limit(20))
    display(df.summary())

# COMMAND ----------

# MAGIC %md ## 4. Nulos por columna

# COMMAND ----------

if table:
    from pyspark.sql import functions as F

    display(df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]))
