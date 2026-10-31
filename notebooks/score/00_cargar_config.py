# Databricks notebook source
# MAGIC %md
# MAGIC # Cargar configuración del score
# MAGIC Lee `config/*.csv` del repo, valida y escribe `config_limits`, `config_weights` y `config_indicators`.
# MAGIC
# MAGIC **Un twist nuevo = filas nuevas en esos CSV** con otro `twist_id` y `vigente_desde`. Por cada clave gana la fila más reciente;
# MAGIC las anteriores se conservan como historial.

# COMMAND ----------

import os
import pandas as pd

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "credicorp_hack")
T = f"{dbutils.widgets.get('catalog')}.{dbutils.widgets.get('schema')}"
CFG = os.path.abspath("../../config")  # el notebook vive en notebooks/score/

KEYS = {
    "config_limits": ["cliente", "regla"],
    "config_weights": ["cliente", "grupo"],
    "config_indicators": ["cliente", "grupo", "indicador"],
}
REGLAS = {"VOL_ALTA_UMBRAL", "VENTANA_SESIONES", "DD_MAX", "PESO_MAX_ALTA_VOL", "NECESIDAD_6M", "P05_6M_MIN"}
INDICADORES = {"dd_base", "vol_base", "p05_6m_base", "ret_anual_base", "razon_vol_base_usd"}

cfg = {}
for name, keys in KEYS.items():
    df = pd.read_csv(f"{CFG}/{name}.csv")
    df["vigente_desde"] = pd.to_datetime(df.vigente_desde).dt.date
    dup = df.duplicated(keys + ["vigente_desde"])
    assert not dup.any(), f"{name}: claves repetidas con la misma vigencia\n{df[dup]}"
    cfg[name] = df

# COMMAND ----------

# MAGIC %md ## Validación (sobre las filas vigentes)

# COMMAND ----------

vig = {n: d.sort_values("vigente_desde").drop_duplicates(KEYS[n], keep="last") for n, d in cfg.items()}
w, ind, lim = vig["config_weights"], vig["config_indicators"], vig["config_limits"]

suma = w.groupby("cliente").peso.sum().round(6)
assert (suma == 1.0).all(), f"los pesos de grupo deben sumar 1 por cliente:\n{suma}"
assert set(zip(w.cliente, w.grupo)) == set(zip(ind.cliente, ind.grupo)), "grupos con peso y grupos con indicadores no coinciden"
assert (ind.ancla_0 != ind.ancla_100).all(), "ancla_0 y ancla_100 no pueden ser iguales"
assert set(ind.indicador) <= INDICADORES, f"indicador desconocido: {set(ind.indicador) - INDICADORES}"
assert set(lim.regla) <= REGLAS, f"regla desconocida: {set(lim.regla) - REGLAS}"
print("Configuración válida")

# COMMAND ----------

for name, df in cfg.items():
    spark.createDataFrame(df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{T}.{name}")
    print(f"{T}.{name}: {len(df)} filas")
display(w)
