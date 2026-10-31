# Databricks notebook source
# MAGIC %md
# MAGIC # Value Finders · 00 · Configurar la API key de Gemini (una sola vez)
# MAGIC Guarda la key en el secret scope `hackathon/gemini_api_key`, que es donde la busca el notebook 02.
# MAGIC
# MAGIC 1. Corre la **celda 1**: aparece arriba un campo `gemini_key`.
# MAGIC 2. **Pega la key en ese campo** (nunca en el código: este notebook está en un repo público).
# MAGIC 3. Corre la **celda 2**: guarda la key y borra el campo.
# MAGIC 4. Corre la **celda 3**: debe imprimir `True`. Luego, en el notebook 02, **Run all**.

# COMMAND ----------

# ===== CELDA 1: CAMPO PARA PEGAR LA KEY =====
dbutils.widgets.text("gemini_key", "")

# COMMAND ----------

# ===== CELDA 2: GUARDAR EN EL SECRET SCOPE =====
from databricks.sdk import WorkspaceClient

key = dbutils.widgets.get("gemini_key").strip()
if not key:
    raise ValueError(
        "El campo gemini_key está vacío: pega la key arriba y vuelve a correr esta celda."
    )

w = WorkspaceClient()
if "hackathon" not in [s.name for s in w.secrets.list_scopes()]:
    w.secrets.create_scope(scope="hackathon")
w.secrets.put_secret(scope="hackathon", key="gemini_api_key", string_value=key)
dbutils.widgets.remove("gemini_key")  # la key no queda en el notebook
print("Guardado:", [s.key for s in w.secrets.list_secrets(scope="hackathon")])

# COMMAND ----------

# ===== CELDA 3: COMPROBAR =====
# Databricks muestra [REDACTED] en lugar del valor: es lo esperado
print(len(dbutils.secrets.get("hackathon", "gemini_api_key")) > 0)
