# Databricks notebook source
# MAGIC %md
# MAGIC # Value Finders · 03 · Linaje Medallion (Bronze → Silver → Gold)
# MAGIC Dibuja el recorrido de los datos con el **estado real** de cada tabla en el catálogo: filas actuales y si ya fue creada.
# MAGIC Córrelo después de `01_bronze_silver` y `02_gold_recomendacion`.
# MAGIC
# MAGIC | Color | Capa | CRISP-DM |
# MAGIC |---|---|---|
# MAGIC | Bronze | CSV tal cual, todo como texto | 2 · Entender los datos |
# MAGIC | Silver | Limpio, cada cambio con regla y porqué | 3 · Preparar los datos |
# MAGIC | Gold | Métricas, reglas, recomendación y evidencia | 4–6 · Modelar, evaluar, desplegar |
# MAGIC | Morado | Agente Gemini: propone reglas o explica; nunca decide pesos | — |

# COMMAND ----------

# ===== CELDA 1: ESTADO DE LAS TABLAS EN EL CATÁLOGO =====
import pandas as pd

dbutils.widgets.text("catalog", "catalogo_hackathon")
dbutils.widgets.text("schema", "hackathon")
DB = f"{dbutils.widgets.get('catalog')}.{dbutils.widgets.get('schema')}"

# Tabla -> (capa, qué contiene). Es el mismo inventario que crean los notebooks 01 y 02.
TABLAS = {
    "bronze_mercado_diario": ("Bronze", "Precio diario por activo"),
    "bronze_mercado_intradia": ("Bronze", "Velas de 10 minutos desde 2020"),
    "bronze_macro": ("Bronze", "Tasa 10Y, inflación, tipos de cambio, régimen"),
    "bronze_fundamentales": ("Bronze", "Estados financieros trimestrales"),
    "bronze_activos": ("Bronze", "Referencia de 8 activos"),
    "bronze_clientes": ("Bronze", "3 perfiles de cliente"),
    "bronze_eventos": ("Bronze", "5 eventos de mercado"),
    "bronze_twist": ("Bronze", "Noticias del 17-ago"),
    "bronze_perfil_archivos": ("Bronze", "Filas, BOM y vacíos por archivo"),
    "silver_mercado_diario": (
        "Silver",
        "Fuente principal de precios, con marcas de calidad",
    ),
    "silver_mercado_intradia": ("Silver", "Intradía limpio"),
    "silver_mercado_intradia_diario": ("Silver", "Intradía agregado a vela diaria"),
    "silver_macro": ("Silver", "Macro con inflación vigente"),
    "silver_fundamentales": ("Silver", "Fundamentales corregidos"),
    "silver_activos": ("Silver", "Referencia de activos"),
    "silver_clientes": ("Silver", "Perfiles tipados"),
    "silver_eventos": ("Silver", "Una fila por activo afectado"),
    "silver_twist": ("Silver", "Twist con cliente y magnitud"),
    "silver_conciliacion_fuentes": ("Silver", "Diario vs. intradía por activo"),
    "silver_coherencia_fuentes": ("Silver", "Qué fuente se mueve con el mercado"),
    "silver_bitacora_calidad": ("Silver", "Cada regla: acción, filas y porqué"),
    "gold_metricas_activo": ("Gold", "Retorno, riesgo, estrés y fundamentales"),
    "gold_impacto_eventos": ("Gold", "Efecto propio de cada evento"),
    "gold_reglas_propuestas": ("Gold", "Reglas que propuso el agente"),
    "gold_reglas_cliente": ("Gold", "Reglas activas que lee el motor"),
    "gold_recomendacion": ("Gold", "Pesos recomendados por cliente"),
    "gold_cumplimiento": ("Gold", "Cada regla: valor, límite y cumple"),
    "gold_backtest": ("Gold", "Portafolio vs. igual peso"),
    "gold_frontera": ("Gold", "Mezclas evaluadas por el motor"),
    "gold_sensibilidad": ("Gold", "Qué supuesto cambia la recomendación"),
    "gold_reglas_twist": ("Gold", "Reglas nuevas por el twist"),
    "gold_recomendacion_twist": ("Gold", "Antes y después del twist"),
    "gold_estres_twist": ("Gold", "Escenarios de tasas y tipo de cambio"),
    "gold_opciones_cliente": ("Gold", "Menú Conservadora / Balanceada / Crecimiento"),
    "gold_opciones_pesos": ("Gold", "Pesos de cada opción"),
    "gold_recomendacion_clases": ("Gold", "Liquidez / renta fija / renta variable por cartera"),
    "gold_justificacion_pesos": ("Gold", "Por qué cada %: aporte, prueba de ±5 puntos y regla que lo fija"),
    "gold_carteras_equipo": ("Gold", "Carteras propuestas por el equipo vs. el motor"),
    "gold_score_activos": ("Gold", "Score 0-100 por activo y cliente: sí / no y por qué"),
    "gold_score_pesos": ("Gold", "Ponderaciones fijas del score por perfil"),
    "gold_brief_cliente": ("Gold", "Brief del agente para el dashboard"),
    "gold_traza_agente": ("Gold", "Pasos del agente explicador"),
}

estado = []
for t, (capa, desc) in TABLAS.items():
    existe = spark.catalog.tableExists(f"{DB}.{t}")
    filas = spark.table(f"{DB}.{t}").count() if existe else None
    estado.append(
        {"capa": capa, "tabla": t, "existe": existe, "filas": filas, "contenido": desc}
    )
df_estado = pd.DataFrame(estado)
FILAS = {e["tabla"]: e["filas"] for e in estado}  # desde la lista: pandas convertiría None en NaN
print(f"{int(df_estado['existe'].sum())} de {len(df_estado)} tablas creadas en {DB}")
display(df_estado)

# COMMAND ----------

# ===== CELDA 2: DIAGRAMA DE LINAJE =====
# Cada nodo muestra las filas actuales; las tablas que aún no existen aparecen punteadas.


def n(t):
    # Etiqueta de un nodo: nombre de la tabla + filas (o "sin crear")
    f = FILAS.get(t)
    return f'{t}["{t}<br/><small>{f"{f:,} filas" if f is not None else "sin crear"}</small>"]'



MERMAID = f"""flowchart LR
  subgraph V["Volumen · CSV"]
    direction TB
    c1["01_market_prices_raw"]
    c1x["01_market_prices_raw_expanded"]
    c2["02_macro_raw"]
    c3["03_fundamentals_raw"]
    c4["04_asset_reference"]
    c5["05_client_profiles"]
    c6["06_events"]
    c7["dataset_twist"]
  end
  subgraph B["BRONZE · notebook 01"]
    direction TB
    {n("bronze_mercado_diario")}
    {n("bronze_mercado_intradia")}
    {n("bronze_macro")}
    {n("bronze_fundamentales")}
    {n("bronze_activos")}
    {n("bronze_clientes")}
    {n("bronze_eventos")}
    {n("bronze_twist")}
  end
  subgraph S["SILVER · notebook 01"]
    direction TB
    {n("silver_mercado_diario")}
    {n("silver_mercado_intradia")}
    {n("silver_mercado_intradia_diario")}
    {n("silver_macro")}
    {n("silver_fundamentales")}
    {n("silver_activos")}
    {n("silver_clientes")}
    {n("silver_eventos")}
    {n("silver_twist")}
    {n("silver_coherencia_fuentes")}
    {n("silver_bitacora_calidad")}
  end
  subgraph G["GOLD · notebook 02"]
    direction TB
    {n("gold_metricas_activo")}
    {n("gold_impacto_eventos")}
    agt(["Agente traductor · Gemini"])
    {n("gold_reglas_propuestas")}
    {n("gold_reglas_cliente")}
    motor{{{{"Motor de recomendación"}}}}
    {n("gold_recomendacion")}
    {n("gold_cumplimiento")}
    {n("gold_backtest")}
    {n("gold_sensibilidad")}
    {n("gold_reglas_twist")}
    {n("gold_recomendacion_twist")}
    {n("gold_estres_twist")}
    {n("gold_opciones_cliente")}
    {n("gold_opciones_pesos")}
    {n("gold_recomendacion_clases")}
    {n("gold_justificacion_pesos")}
    {n("gold_carteras_equipo")}
    {n("gold_score_activos")}
    eq[/"Carteras del equipo · widget"/]
    agx(["Agente explicador · Gemini"])
    {n("gold_brief_cliente")}
  end
  dash[["Decision Dashboard"]]

  c1 --> bronze_mercado_diario --> silver_mercado_diario
  c1x --> bronze_mercado_intradia --> silver_mercado_intradia --> silver_mercado_intradia_diario
  c2 --> bronze_macro --> silver_macro
  c3 --> bronze_fundamentales --> silver_fundamentales
  c4 --> bronze_activos --> silver_activos
  c5 --> bronze_clientes --> silver_clientes
  c6 --> bronze_eventos --> silver_eventos
  c7 --> bronze_twist --> silver_twist
  silver_mercado_diario --> silver_coherencia_fuentes
  silver_mercado_intradia_diario --> silver_coherencia_fuentes
  silver_macro --> silver_coherencia_fuentes
  silver_mercado_diario -. "reglas de limpieza" .-> silver_bitacora_calidad
  silver_fundamentales -.-> silver_bitacora_calidad
  silver_macro -.-> silver_bitacora_calidad

  silver_mercado_diario -- "fuente_precios = diario" --> gold_metricas_activo
  silver_macro --> gold_metricas_activo
  silver_fundamentales --> gold_metricas_activo
  silver_activos --> gold_metricas_activo
  silver_eventos --> gold_impacto_eventos
  silver_clientes --> agt --> gold_reglas_propuestas --> gold_reglas_cliente
  gold_reglas_cliente --> motor
  silver_mercado_diario --> motor
  silver_macro -- "tipo de cambio" --> motor
  gold_metricas_activo -- "papel ancla / motor" --> motor
  motor --> gold_recomendacion
  motor --> gold_cumplimiento
  motor --> gold_backtest
  motor --> gold_sensibilidad
  silver_twist --> gold_reglas_twist --> motor
  motor --> gold_recomendacion_twist
  motor --> gold_estres_twist
  motor -- "3 objetivos" --> gold_opciones_cliente
  gold_opciones_cliente --> agx
  silver_twist -- "escenarios tasas / FX" --> gold_estres_twist
  gold_reglas_twist --> gold_opciones_cliente
  gold_opciones_cliente --> gold_opciones_pesos
  gold_opciones_pesos --> gold_recomendacion_clases
  gold_recomendacion --> gold_recomendacion_clases
  gold_recomendacion_clases --> dash
  gold_recomendacion --> gold_justificacion_pesos
  gold_opciones_cliente --> gold_justificacion_pesos
  eq --> gold_carteras_equipo
  gold_reglas_twist --> gold_carteras_equipo
  gold_carteras_equipo --> gold_justificacion_pesos
  gold_justificacion_pesos --> agx
  gold_carteras_equipo --> agx
  gold_justificacion_pesos --> dash
  gold_carteras_equipo --> dash
  gold_metricas_activo -- "6 componentes" --> gold_score_activos
  gold_impacto_eventos --> gold_score_activos
  gold_estres_twist --> gold_score_activos
  gold_opciones_cliente --> gold_score_activos
  gold_score_activos --> agx
  gold_score_activos --> dash
  silver_macro -- "tasa 10Y: bono sintético" --> motor
  gold_opciones_cliente --> dash
  gold_recomendacion --> agx
  gold_cumplimiento --> agx
  gold_sensibilidad --> agx
  gold_estres_twist --> agx
  silver_bitacora_calidad --> agx
  agx --> gold_brief_cliente
  gold_recomendacion --> dash
  gold_cumplimiento --> dash
  gold_recomendacion_twist --> dash
  gold_estres_twist --> dash
  gold_brief_cliente --> dash

  classDef csv fill:#eef0f4,stroke:#8a93a6,color:#1c2230
  classDef bronze fill:#f3e4d6,stroke:#9a5b2e,color:#1c2230
  classDef silver fill:#e6e9ef,stroke:#5f6b7d,color:#1c2230
  classDef gold fill:#f6ecc4,stroke:#8a6a00,color:#1c2230
  classDef agente fill:#e5e6fb,stroke:#4b4fb8,color:#1c2230
  classDef motorc fill:#fff6d6,stroke:#8a6a00,stroke-width:2px,color:#1c2230
  classDef dashc fill:#1c2230,stroke:#1c2230,color:#ffffff
  classDef falta stroke-dasharray:5 4,opacity:0.55
  class c1,c1x,c2,c3,c4,c5,c6,c7 csv
  class {",".join(t for t in TABLAS if t.startswith("bronze_") and t != "bronze_perfil_archivos")} bronze
  class {",".join(t for t in TABLAS if t.startswith("silver_") and t != "silver_conciliacion_fuentes")} silver
  class {",".join(t for t in TABLAS if t.startswith("gold_") and t not in ("gold_frontera", "gold_traza_agente"))} gold
  class agt,agx agente
  class motor motorc
  class dash dashc
"""
# Las tablas que faltan se dibujan punteadas (solo las que aparecen como nodo en el diagrama)
nodos = [t for t in TABLAS if f"{t}[" in MERMAID]
pendientes = [t for t in nodos if FILAS.get(t) is None]
if pendientes:
    MERMAID += f"  class {','.join(pendientes)} falta\n"

displayHTML(f"""
<div style="font-family: system-ui, sans-serif; overflow-x: auto;">
  <pre class="mermaid" style="min-width: 1100px;">{MERMAID}</pre>
</div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
<script>
  mermaid.initialize({{ startOnLoad: false, securityLevel: "loose", flowchart: {{ htmlLabels: true, curve: "basis" }} }});
  mermaid.run();
</script>
""")

# COMMAND ----------

# ===== CELDA 3: CÓDIGO MERMAID (para pegar en docs o slides) =====
print(MERMAID)
