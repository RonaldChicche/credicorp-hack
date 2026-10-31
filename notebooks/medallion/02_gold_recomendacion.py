# Databricks notebook source
# MAGIC %md
# MAGIC # Value Finders · 02 · Gold: métricas → reglas → recomendación → evidencia
# MAGIC **Pregunta:** ¿Qué decisión de inversión recomendarían a este cliente hoy, a partir de la información disponible, y qué evidencia les permite defenderla?
# MAGIC
# MAGIC | CRISP-DM | Celda | Tabla Gold |
# MAGIC |---|---|---|
# MAGIC | 4. Modelado (insumos) | 1–3 · Métricas por activo, fundamentales, eventos | `gold_metricas_activo`, `gold_impacto_eventos` |
# MAGIC | 1. Negocio → reglas | 4–5 · Perfil del cliente a reglas medibles (agente propone, código valida) | `gold_reglas_cliente`, `gold_reglas_propuestas` |
# MAGIC | 4. Modelado | 6 · Motor genérico: mezcla que cumple todas las reglas | `gold_recomendacion`, `gold_cumplimiento`, `gold_backtest`, `gold_frontera` |
# MAGIC | 5. Evaluación | 7 · Sensibilidad: ¿qué supuesto cambia la recomendación? | `gold_sensibilidad` |
# MAGIC | 6. Despliegue | 8 · Agente explicador: brief del cliente elegido citando tablas | `gold_brief_cliente`, `gold_traza_agente` |
# MAGIC
# MAGIC **División de roles:** el agente traduce y explica; el motor decide los pesos. El agente nunca elige pesos.

# COMMAND ----------

# MAGIC %pip install google-genai

# COMMAND ----------

# ===== CELDA 0: CONFIGURACIÓN =====
import json, re
import numpy as np
import pandas as pd
from datetime import datetime

dbutils.widgets.text("catalog", "catalogo_hackathon")
dbutils.widgets.text("schema", "hackathon")
dbutils.widgets.text(
    "fuente_precios", "diario"
)  # diario | intradia: el diario es el coherente con macro (ver silver_coherencia_fuentes)
dbutils.widgets.text("cliente_elegido", "CL_02")
dbutils.widgets.text("brief_clientes", "todos")  # "todos" o lista: CL_01,CL_03
dbutils.widgets.text(
    "auto_aprobar_reglas", "si"
)  # "no" deja las reglas del agente en estado propuesta
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
FUENTE = dbutils.widgets.get("fuente_precios").strip().lower()
CLIENTE_ELEGIDO = dbutils.widgets.get("cliente_elegido").strip()
BRIEF_CLIENTES = dbutils.widgets.get("brief_clientes").strip()
# Carteras que propone el equipo para comparar con el motor (JSON). Claves aceptadas: código (VF_A05),
# nombre (Aurora Health), LIQUIDEZ (caja en la moneda del cliente), CAJA_USD, BONO. Ejemplo de un compañero:
dbutils.widgets.text(
    "carteras_equipo",
    '{"Equipo: 35% liquidez, 25% Aurora": {"cliente": "CL_02", "pesos": '
    '{"LIQUIDEZ": 0.35, "Aurora Health": 0.25, "Lima Financial": 0.30, "BONO": 0.10}}}',
)
CARTERAS_EQUIPO = dbutils.widgets.get("carteras_equipo").strip()
AUTO_APROBAR = dbutils.widgets.get("auto_aprobar_reglas").strip().lower() == "si"
DB = f"{CATALOG}.{SCHEMA}"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

VENTANA = 252  # 1 año bursátil hasta la fecha de corte
UMBRAL_ALTA_VOL = 0.30  # supuesto: "alta volatilidad" = vol anual > 30%
BONO = "BONO_UST10"  # renta fija: bono del Tesoro de EE. UU. a 10 años, construido desde us_10y_yield
ESTRES = ("2026-02-09", "2026-02-20")  # único periodo risk_regime = Stress
DIAS_PUBLICACION = 45  # un trimestre se conoce ~45 días después de su cierre
N_CANDIDATOS = 30000
PASO_PESO = 0.05  # pesos presentables: múltiplos de 5%
SEMILLA = 42
dbutils.widgets.text("modelo_gemini", "gemini-3.5-flash-lite")
MODELO = dbutils.widgets.get("modelo_gemini").strip()
dbutils.widgets.text(
    "objetivo_desde_agente", "no"
)  # "si": el agente también elige qué se maximiza
OBJETIVO_AGENTE = dbutils.widgets.get("objetivo_desde_agente").strip().lower() == "si"
SECRET_SCOPE, SECRET_KEY = "hackathon", "gemini_api_key"


def guardar(df, nombre):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object and df[c].isna().all():
            df[c] = ""
    spark.createDataFrame(df).write.mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(f"{DB}.{nombre}")


def leer(nombre):
    return spark.table(f"{DB}.{nombre}").toPandas()


# Gemini: la key vive en un secret scope, nunca en el código (el repo es público).
# Se verifica paso a paso para que, si falla, el mensaje diga exactamente qué falta.
cliente_llm, AGENTE_ACTIVO = None, False


def preguntar(prompt):
    return cliente_llm.models.generate_content(model=MODELO, contents=prompt).text


try:
    from google import genai
except ImportError:
    MOTIVO_AGENTE = (
        "falta la librería: corre la celda '%pip install google-genai' y luego Run all."
    )
else:
    try:
        api_key = dbutils.secrets.get(SECRET_SCOPE, SECRET_KEY)
    except Exception as e:
        api_key = None
        MOTIVO_AGENTE = (
            f"no se pudo leer el secret {SECRET_SCOPE}/{SECRET_KEY} ({type(e).__name__}: {str(e)[:200]}). "
            "Créalo con el notebook 00_configurar_gemini y luego Run all."
        )
    if api_key:
        try:
            cliente_llm = genai.Client(api_key=api_key)
            preguntar(
                "Responde solo: ok"
            )  # prueba real: valida key y nombre del modelo
            AGENTE_ACTIVO = True
            MOTIVO_AGENTE = (
                f"modelo {MODELO}, key desde el secret {SECRET_SCOPE}/{SECRET_KEY}"
            )
        except Exception as e:
            cliente_llm = None
            MOTIVO_AGENTE = (
                f"el secret existe pero Gemini rechazó la llamada ({type(e).__name__}: {str(e)[:300]}). "
                "Revisa que la key sea válida y el widget modelo_gemini."
            )
print(("Agente ACTIVO: " if AGENTE_ACTIVO else "Agente desactivado: ") + MOTIVO_AGENTE)


def extraer_json(texto, abre="[", cierra="]"):
    return json.loads(texto[texto.find(abre) : texto.rfind(cierra) + 1])


# COMMAND ----------

# ===== CELDA 1: PANEL DE PRECIOS, MACRO Y MONEDAS =====
activos = leer("silver_activos")
INFO = activos.set_index("asset_id")
INFO.loc[BONO, ["name", "sector", "country", "currency"]] = [
    "Bono del Tesoro EE. UU. 10 años",
    "Renta fija",
    "United States",
    "USD",
]
macro = leer("silver_macro")
macro["fecha"] = pd.to_datetime(macro["fecha"]).dt.normalize()
macro = macro.set_index("fecha").sort_index()


def retorno_bono(indice):
    # Retorno diario de un bono a 10 años a partir de su tasa (no hay precios de bonos en el caso):
    # cupón diario (tasa del día anterior / 252) menos duración × cambio de tasa.
    # Si la tasa sube, el precio del bono cae; si baja, sube.
    y = (macro["us_10y_yield"] / 100).reindex(
        indice, method="ffill"
    )  # usa el último dato macro conocido
    return (y.shift(1) / 252 - DURACION_BONO * y.diff()).iloc[1:].fillna(0.0)


def duracion_modificada(tasa, anios=10, pagos=2):
    # Duración de un bono a la par con cupón semestral, calculada con la tasa del propio CSV (no es un supuesto):
    # cuánto cae su precio (%) por cada 1 punto que sube la tasa.
    i, n = tasa / pagos, anios * pagos
    macaulay = (1 + i) / i * (1 - (1 + i) ** -n) / pagos
    return macaulay / (1 + i)


# Con la tasa a la fecha de corte (02_macro_raw.csv)
DURACION_BONO = duracion_modificada(macro["us_10y_yield"].dropna().iloc[-1] / 100)
print(
    f"Duración del bono a 10 años: {DURACION_BONO:.2f} (tasa {macro['us_10y_yield'].dropna().iloc[-1]:.2f}%)"
)


def panel_precios(fuente):
    # Cierre diario por activo desde Silver, sin filas excluidas por calidad
    if fuente == "intradia":
        px = leer("silver_mercado_intradia_diario")
    else:
        px = leer("silver_mercado_diario")
        px = px[~px["dq_excluir"].astype(bool)]
    px["fecha"] = pd.to_datetime(px["fecha"]).dt.normalize()
    close = px.pivot_table(
        index="fecha", columns="asset_id", values="close", aggfunc="last"
    ).sort_index()
    corte = min(close.index.max(), macro.index.max())
    # Un día sin precio para un activo repite el último cierre (retorno 0 ese día) en vez de borrar el día
    return close.loc[:corte].ffill(), corte


close_hist, FECHA_CORTE = panel_precios(FUENTE)
close = close_hist.iloc[-(VENTANA + 1) :]
ret = close.pct_change().iloc[1:]
print(
    f"Fuente: {FUENTE} | corte ('hoy'): {FECHA_CORTE.date()} | ventana: {ret.index.min().date()} → {ret.index.max().date()}"
)

# Tasa libre de riesgo: rendimiento del bono de EE. UU. a 10 años (anual, en %)
rf_diaria = (macro["us_10y_yield"] / 100 / 252).reindex(close.index).ffill()
RF_ANUAL = float(macro.loc[ret.index.min() : FECHA_CORTE, "us_10y_yield"].mean() / 100)
FX_COL = {"COP": "usd_cop", "PEN": "usd_pen", "MXN": "usd_mxn", "CLP": "usd_clp"}


def retornos_cliente(close_panel, moneda):
    # Retornos diarios en la moneda del cliente. CAJA = depósito en USD que rinde la tasa libre de riesgo.
    # Un inversionista en COP que tiene un activo en USD gana (o pierde) también por el tipo de cambio.
    r = close_panel.pct_change().iloc[1:].copy()
    r["CAJA"] = (macro["us_10y_yield"] / 100 / 252).reindex(r.index).ffill().values
    r[BONO] = retorno_bono(close_panel.index).reindex(r.index).fillna(0.0)
    if moneda != "USD":
        fx = (
            macro[FX_COL[moneda]]
            .reindex(close_panel.index)
            .ffill()
            .pct_change()
            .iloc[1:]
        )
        r = (1 + r).mul(1 + fx, axis=0) - 1
        # CAJA_LOCAL = liquidez en la moneda del cliente, sin riesgo cambiario. El caso no trae una tasa
        # local (COP/PEN), así que rinde 0%: supuesto conservador, declarado en gold_reglas_cliente.
        r["CAJA_LOCAL"] = 0.0
    return r


def es_caja(a):
    return a in ("CAJA", "CAJA_LOCAL")


def clase(a):
    # Clase de activo, como la presenta un asesor
    return "Liquidez" if es_caja(a) else "Renta fija" if a == BONO else "Renta variable"


# COMMAND ----------


# ===== CELDA 2: MÉTRICAS POR ACTIVO (gold_metricas_activo) =====
def max_drawdown(r):
    valor = np.cumprod(1 + np.asarray(r))
    pico = np.maximum.accumulate(np.maximum(valor, 1.0))
    return float((valor / pico - 1).min())


mf = macro["market_factor"].reindex(ret.index)
ini_e, fin_e = pd.Timestamp(ESTRES[0]), pd.Timestamp(ESTRES[1])
filas = []
for a in ret.columns:
    r = ret[a].dropna()
    vol = r.std() * np.sqrt(252)
    antes = close_hist.loc[: ini_e - pd.Timedelta(days=1), a].iloc[-1]
    durante = close_hist.loc[:fin_e, a].iloc[-1]
    rh = close_hist[a].pct_change().dropna()
    filas.append(
        {
            "asset_id": a,
            "retorno_12m": close[a].iloc[-1] / close[a].iloc[0] - 1,
            "retorno_3m": close[a].iloc[-1] / close[a].iloc[-64] - 1,
            "vol_anual": vol,
            "max_drawdown": max_drawdown(r),
            "sharpe": (r.mean() * 252 - RF_ANUAL) / vol,
            "beta_mercado": r.cov(mf) / mf.var(),
            "retorno_estres": durante / antes - 1,
            # Robustez: el mismo activo medido en toda la historia disponible (intradía: desde 2020)
            "vol_historica": rh.std() * np.sqrt(252),
            "max_drawdown_historico": max_drawdown(rh),
            "historia_desde": close_hist[a].first_valid_index(),
        }
    )
# Renta fija: mismas métricas para el bono, calculadas con su retorno diario desde la tasa
rb = retorno_bono(close.index)
rb_hist = retorno_bono(close_hist.index)
valor_b = (1 + rb_hist).cumprod()
filas.append(
    {
        "asset_id": BONO,
        "retorno_12m": float((1 + rb).prod() - 1),
        "retorno_3m": float((1 + rb.iloc[-63:]).prod() - 1),
        "vol_anual": rb.std() * np.sqrt(252),
        "max_drawdown": max_drawdown(rb),
        "sharpe": (rb.mean() * 252 - RF_ANUAL) / (rb.std() * np.sqrt(252)),
        "beta_mercado": rb.cov(mf) / mf.var(),
        "retorno_estres": float((1 + rb.loc[ini_e:fin_e]).prod() - 1),
        "vol_historica": rb_hist.std() * np.sqrt(252),
        "max_drawdown_historico": max_drawdown(rb_hist),
        "historia_desde": rb_hist.index.min(),
    }
)
met = pd.DataFrame(filas)
met["es_alta_vol"] = met["vol_anual"] > UMBRAL_ALTA_VOL
vol_med = met["vol_anual"].median()
met["papel"] = np.where(
    (met["vol_anual"] <= vol_med) & (met["beta_mercado"] < 1), "ancla", "motor"
)

# Fundamentales: solo trimestres que ya se conocían en la fecha de corte (sin mirar al futuro)
fund = leer("silver_fundamentales")
fund = fund[~fund["dq_excluir"].astype(bool)].copy()
fund["periodo"] = pd.to_datetime(fund["periodo"])
fund = fund[
    fund["periodo"] <= FECHA_CORTE - pd.Timedelta(days=DIAS_PUBLICACION)
].sort_values(["issuer_id", "periodo"])
fund["cash_usd_m"] = fund.groupby("issuer_id")[
    "cash_usd_m"
].ffill()  # caja vacía: último dato conocido


def ttm(g):
    u, p = g.tail(4), g.iloc[-8:-4]
    completo = len(g) >= 8
    return pd.Series(
        {
            "ultimo_trimestre": g["periodo"].iloc[-1],
            "margen_ebitda": u["ebitda_usd_m"].sum() / u["revenue_usd_m"].sum(),
            "deuda_neta_ebitda": (g["debt_usd_m"].iloc[-1] - g["cash_usd_m"].iloc[-1])
            / u["ebitda_usd_m"].sum(),
            "crec_ventas_12m": u["revenue_usd_m"].sum() / p["revenue_usd_m"].sum() - 1
            if completo
            else np.nan,
            "crec_utilidad_12m": u["net_income_usd_m"].sum()
            / p["net_income_usd_m"].sum()
            - 1
            if completo
            else np.nan,
        }
    )


fund_ttm = fund.groupby("issuer_id").apply(ttm).reset_index()
met = met.merge(
    INFO.reset_index()[["asset_id", "issuer_id", "name", "sector", "country"]],
    on="asset_id",
    how="left",
).merge(fund_ttm, on="issuer_id", how="left")
met["clase"] = met["asset_id"].map(clase)
met["fuente_precios"], met["fecha_corte"], met["run_id"] = FUENTE, FECHA_CORTE, RUN_ID
guardar(met, "gold_metricas_activo")
display(met.round(3))

# COMMAND ----------

# ===== CELDA 3: IMPACTO DE EVENTOS (gold_impacto_eventos) =====
# Retorno del activo desde el cierre previo al evento hasta 5 sesiones después, contra el promedio
# de todos los activos en la misma ventana: lo que queda es el efecto "propio" del evento.
ev = leer("silver_eventos")
ev["fecha"] = pd.to_datetime(ev["fecha"])
filas = []
for e in ev.itertuples():
    i = close_hist.index.searchsorted(e.fecha)
    if i == 0 or i >= len(close_hist) or e.asset_id not in close_hist:
        continue
    base, t5 = close_hist.iloc[i - 1], close_hist.iloc[min(i + 4, len(close_hist) - 1)]
    r_act = t5[e.asset_id] / base[e.asset_id] - 1
    r_mkt = (t5 / base - 1).mean()
    filas.append(
        {
            "fecha": e.fecha,
            "asset_id": e.asset_id,
            "headline": e.headline,
            "severity": e.severity,
            "event_type": e.event_type,
            "ret_5d": r_act,
            "ret_5d_mercado": r_mkt,
            "efecto_propio_5d": r_act - r_mkt,
            "ret_hasta_corte": close_hist[e.asset_id].iloc[-1] / base[e.asset_id] - 1,
        }
    )
impacto = pd.DataFrame(filas)
guardar(impacto, "gold_impacto_eventos")
display(impacto.round(3))

# COMMAND ----------

# ===== CELDA 4: CATÁLOGO DE REGLAS Y REGLAS BASE =====
# El motor solo entiende estos tipos. Un cliente nuevo = filas nuevas en gold_reglas_cliente, no código nuevo.
# Rango numérico = (mín, máx) permitido; lista = valores permitidos.
CATALOGO = {
    "DD_MAX": {"limite": (0.01, 1.0)},
    "VOL_MAX": {"limite": (0.01, 1.0)},
    "MAX_POR_SECTOR": {"limite": (0.05, 1.0)},
    "MAX_POR_PAIS": {"limite": (0.05, 1.0)},
    "MAX_GRUPO": {"grupo": ["alta_vol"], "limite": (0.0, 1.0)},
    "MIN_PAISES": {"n": (1, 8), "peso_min": (0.0, 0.5)},
    "MIN_SECTORES": {"n": (1, 8), "peso_min": (0.0, 0.5)},
    "MAX_PESO_ACTIVO": {"limite": (0.05, 1.0)},
    "MIN_CAJA": {"limite": (0.0, 1.0)},
    "MAX_CAJA": {"limite": (0.0, 1.0)},
    "MIN_CAJA_LOCAL": {"limite": (0.0, 1.0)},
    "EXCLUIR_ACTIVO": {"asset_id": sorted(INFO.index)},
    "OBJETIVO": {"tipo": ["max_sharpe", "max_retorno", "min_volatilidad"]},
}
DESCRIPCION = {
    "DD_MAX": "caída máxima del portafolio (fracción)",
    "VOL_MAX": "volatilidad anual máxima del portafolio",
    "MAX_POR_SECTOR": "peso máximo en un mismo sector",
    "MAX_POR_PAIS": "peso máximo en un mismo país",
    "MAX_GRUPO": "peso máximo sumado en activos de alta volatilidad",
    "MIN_PAISES": "mínimo de países con peso ≥ peso_min",
    "MIN_SECTORES": "mínimo de sectores con peso ≥ peso_min",
    "MAX_PESO_ACTIVO": "peso máximo en un solo activo",
    "MIN_CAJA": "liquidez mínima (CAJA)",
    "MAX_CAJA": "liquidez máxima (CAJA): obliga a estar invertido",
    "MIN_CAJA_LOCAL": "liquidez mínima en la moneda del cliente, sin riesgo cambiario (CAJA_LOCAL)",
    "EXCLUIR_ACTIVO": "activo prohibido",
    "OBJETIVO": "qué se maximiza dentro de las reglas",
}


def validar(rule_type, params):
    # Devuelve (ok, motivo). El modelo puede proponer cualquier cosa; aquí se decide qué entra.
    if rule_type not in CATALOGO:
        return False, f"tipo {rule_type} fuera del catálogo"
    esquema = CATALOGO[rule_type]
    if not isinstance(params, dict) or set(params) != set(esquema):
        return False, f"parámetros esperados {sorted(esquema)}"
    for k, regla in esquema.items():
        v = params[k]
        if isinstance(regla, list) and v not in regla:
            return False, f"{k}={v} no permitido"
        if isinstance(regla, tuple):
            if not isinstance(v, (int, float)) or not regla[0] <= v <= regla[1]:
                return False, f"{k}={v} fuera de {regla}"
    return True, "ok"


# Reglas que salen directo de columnas numéricas del perfil: no necesitan interpretación
clientes = leer("silver_clientes")
base = []
for c in clientes.itertuples():
    base.append(
        {
            "client_id": c.client_id,
            "rule_type": "MONEDA_BASE",
            "params": {"moneda": c.base_currency},
            "origen": "perfil",
            "evidencia_texto": f"base_currency = {c.base_currency}",
            "supuesto": False,
        }
    )
    base.append(
        {
            "client_id": c.client_id,
            "rule_type": "DD_MAX",
            "params": {"limite": float(c.max_drawdown)},
            "origen": "perfil",
            "evidencia_texto": f"max_drawdown_tolerance_pct = {c.max_drawdown_tolerance_pct}",
            "supuesto": False,
        }
    )

# Reglas de respaldo GENERADAS desde el perfil (no escritas a mano por cliente): un cliente nuevo en
# 05_client_profiles.csv funciona aunque el agente esté apagado. También sirven para comparar con el agente.
# Políticas de la casa (supuestos declarados): cuánta caja exige cada nivel de liquidez y cuánta
# caja máxima admite cada tolerancia al riesgo, para que el cliente esté invertido.
MIN_CAJA_POR_LIQUIDEZ = {"baja": 0.0, "media": 0.10, "alta": 0.20}
MAX_CAJA_POR_RIESGO = {"alta": 0.10, "media": 0.30, "media-baja": 0.60, "baja": 0.80}
PATRONES_RESTRICCION = [
    # (regex sobre el texto de "constraint", función que arma la regla)
    (
        r"m[aá]x\.?\s*(\d+)\s*%\s*en un solo sector",
        lambda m: ("MAX_POR_SECTOR", {"limite": int(m[1]) / 100}),
    ),
    (
        r"m[aá]x\.?\s*(\d+)\s*%\s*en un solo pa[ií]s",
        lambda m: ("MAX_POR_PAIS", {"limite": int(m[1]) / 100}),
    ),
    (
        r"m[aá]x\.?\s*(\d+)\s*%\s*en (?:un solo activo|una sola empresa)",
        lambda m: ("MAX_PESO_ACTIVO", {"limite": int(m[1]) / 100}),
    ),
    (
        r"m[aá]x\.?\s*(\d+)\s*%\s*en activos de alta volatilidad",
        lambda m: ("MAX_GRUPO", {"grupo": "alta_vol", "limite": int(m[1]) / 100}),
    ),
    (
        r"al menos\s*(\d+)\s*pa[ií]ses",
        lambda m: ("MIN_PAISES", {"n": int(m[1]), "peso_min": 0.05}),
    ),
    (
        r"(\d+)\s*sectores",
        lambda m: ("MIN_SECTORES", {"n": int(m[1]), "peso_min": 0.05}),
    ),
]


def reglas_desde_perfil(c):
    reglas_c = []
    texto = str(c.constraint)
    for patron, armar_regla in PATRONES_RESTRICCION:
        m = re.search(patron, texto, re.I)
        if m:
            t, p = armar_regla(m)
            reglas_c.append((t, p, m.group(0), False))
    if not reglas_c and texto.strip():
        print(
            f"{c.client_id}: restricción no reconocida '{texto}'. Agrégala a PATRONES_RESTRICCION o deja que la traduzca el agente."
        )
    liquidez = str(c.liquidity_need).strip().lower()
    riesgo = str(c.risk_tolerance).strip().lower()
    if liquidez not in MIN_CAJA_POR_LIQUIDEZ or riesgo not in MAX_CAJA_POR_RIESGO:
        print(
            f"{c.client_id}: liquidez '{c.liquidity_need}' o riesgo '{c.risk_tolerance}' sin política; se usa la de nivel 'media'."
        )
        liquidez = liquidez if liquidez in MIN_CAJA_POR_LIQUIDEZ else "media"
        riesgo = riesgo if riesgo in MAX_CAJA_POR_RIESGO else "media"
    if MIN_CAJA_POR_LIQUIDEZ[liquidez] > 0:
        reglas_c.append(
            (
                "MIN_CAJA",
                {"limite": MIN_CAJA_POR_LIQUIDEZ[liquidez]},
                f"Liquidez {c.liquidity_need}",
                True,
            )
        )
    reglas_c.append(
        (
            "MAX_CAJA",
            {"limite": MAX_CAJA_POR_RIESGO[riesgo]},
            f"Tolerancia {c.risk_tolerance}: se mantiene invertido",
            True,
        )
    )
    prioridad = str(c.priority).lower()
    objetivo = "max_sharpe" if "diversific" in prioridad else "max_retorno"
    reglas_c.append(("OBJETIVO", {"tipo": objetivo}, c.priority, False))
    return reglas_c


# Un cliente con moneda sin tipo de cambio en 02_macro no se puede medir en su moneda: se omite con aviso
clientes["base_currency"] = clientes["base_currency"].str.strip().str.upper()
soportada = clientes["base_currency"].isin(["USD", *FX_COL])
for c in clientes[~soportada].itertuples():
    print(
        f"OMITIDO {c.client_id}: moneda {c.base_currency} sin columna de tipo de cambio en 02_macro (FX_COL)."
    )
clientes = clientes[soportada].reset_index(drop=True)
MANUALES = {c.client_id: reglas_desde_perfil(c) for c in clientes.itertuples()}
for cid, rr in MANUALES.items():
    print(cid, [(t, p) for t, p, _, _ in rr])

# COMMAND ----------

# ===== CELDA 5: AGENTE TRADUCTOR (perfil en texto → reglas del catálogo) =====
# El agente propone; el código valida contra el catálogo; un humano (o auto_aprobar) aprueba.
catalogo_txt = "\n".join(
    f"- {k}: params {json.dumps({p: (list(v) if isinstance(v, tuple) else v) for p, v in e.items()}, ensure_ascii=False)} · {DESCRIPCION[k]}"
    for k, e in CATALOGO.items()
)
propuestas = []
for c in clientes.itertuples():
    perfil = {
        k: getattr(c, k)
        for k in [
            "client_id",
            "profile_name",
            "horizon_months",
            "risk_tolerance",
            "liquidity_need",
            "priority",
            "constraint",
        ]
    }
    filas_agente = []
    if AGENTE_ACTIVO:
        prompt = f"""Eres analista de riesgo de una gestora de inversiones. Traduce el perfil del cliente a reglas.
Usa SOLO estos tipos y parámetros (rango = [mín, máx]; lista = valores permitidos):
{catalogo_txt}

Reglas:
- No generes MONEDA_BASE ni DD_MAX: ya salen de columnas numéricas.
- Exactamente un OBJETIVO. Criterio: "crecimiento" -> max_retorno; "preservación + crecimiento moderado" -> max_retorno
  (la preservación ya la garantiza DD_MAX); "diversificación" -> max_sharpe. Usa min_volatilidad solo si el perfil
  pide literalmente minimizar el riesgo: con caja disponible, min_volatilidad deja todo en caja y no es una recomendación.
- Propón siempre un MAX_CAJA (supuesto: true) acorde al perfil: el cliente busca invertir, no quedarse en efectivo.
- Traduce lo literal de "constraint" (ej. "Máx. 45% en un solo sector" -> MAX_POR_SECTOR limite 0.45).
- Para "liquidity_need", "priority" y "risk_tolerance" puedes proponer reglas de CAJA u OBJETIVO;
  si el número no está escrito en el perfil, marca "supuesto": true.
- "evidencia_texto" copia el fragmento del perfil que justifica la regla.
Responde solo una lista JSON:
[{{"rule_type": "", "params": {{}}, "evidencia_texto": "", "supuesto": false}}]

PERFIL: {json.dumps(perfil, ensure_ascii=False, default=str)}"""
        try:
            filas_agente = extraer_json(preguntar(prompt))
        except Exception as e:
            print(c.client_id, "respuesta no válida del agente:", e)
    for r in filas_agente:
        ok, motivo = validar(r.get("rule_type"), r.get("params"))
        if r.get("rule_type") in ("MONEDA_BASE", "DD_MAX"):
            ok, motivo = False, "viene del perfil numérico"
        estado = ("aprobada" if AUTO_APROBAR else "propuesta") if ok else "rechazada"
        propuestas.append(
            {
                "client_id": c.client_id,
                "rule_type": r.get("rule_type"),
                "params": r.get("params"),
                "origen": "agente",
                "evidencia_texto": r.get("evidencia_texto", ""),
                "supuesto": bool(r.get("supuesto", False)),
                "estado": estado,
                "validacion": motivo,
            }
        )
    for t, p, ev_txt, sup in MANUALES.get(c.client_id, []):
        propuestas.append(
            {
                "client_id": c.client_id,
                "rule_type": t,
                "params": p,
                "origen": "manual",
                "evidencia_texto": ev_txt,
                "supuesto": sup,
                "estado": "respaldo",
                "validacion": "ok",
            }
        )

df_prop = pd.DataFrame(propuestas)
df_prop["params"] = df_prop["params"].map(lambda p: json.dumps(p, ensure_ascii=False))
df_prop["run_id"] = RUN_ID
guardar(df_prop, "gold_reglas_propuestas")

# Reglas activas: perfil + agente aprobadas; si el agente no dejó un OBJETIVO válido, se usan las manuales
activas = []
for cid in clientes["client_id"]:
    ag = [
        p
        for p in propuestas
        if p["client_id"] == cid
        and p["origen"] == "agente"
        and p["estado"] == "aprobada"
    ]
    # Una regla activa por tipo (EXCLUIR_ACTIVO, una por activo): si el agente repite un tipo, gana la primera
    vistas, unicas = set(), []
    for p in ag:
        clave = (
            p["rule_type"],
            p["params"].get("asset_id") if p["rule_type"] == "EXCLUIR_ACTIVO" else None,
        )
        if clave not in vistas:
            vistas.add(clave)
            unicas.append(p)
    ag = unicas
    manual = [
        p for p in propuestas if p["client_id"] == cid and p["origen"] == "manual"
    ]
    if len([p for p in ag if p["rule_type"] == "OBJETIVO"]) == 1:
        # Cobertura obligatoria: si el agente no cubrió un tipo de regla que sí tiene la versión manual,
        # entra la manual. Así nunca se pierden reglas literales ni las barandas de negocio
        # (sin MAX_CAJA, un objetivo de baja volatilidad terminaría 100% en caja)
        # El OBJETIVO es una decisión de negocio: por defecto manda el del equipo y el del agente
        # queda solo como propuesta en gold_reglas_propuestas (widget objetivo_desde_agente = si para usarlo)
        if not OBJETIVO_AGENTE:
            ag = [p for p in ag if p["rule_type"] != "OBJETIVO"]
        cubiertas = {p["rule_type"] for p in ag}
        usar = ag + [p for p in manual if p["rule_type"] not in cubiertas]
    else:
        usar = manual
    activas += [b for b in base if b["client_id"] == cid] + usar
reglas = pd.DataFrame(activas)[
    ["client_id", "rule_type", "params", "origen", "evidencia_texto", "supuesto"]
]
reglas_tbl = reglas.assign(
    params=reglas["params"].map(lambda p: json.dumps(p, ensure_ascii=False)),
    run_id=RUN_ID,
)
guardar(reglas_tbl, "gold_reglas_cliente")
display(reglas_tbl)

# COMMAND ----------

# ===== CELDA 6: MOTOR GENÉRICO (gold_recomendacion, gold_cumplimiento, gold_backtest, gold_frontera) =====
# Prueba miles de mezclas, se queda con las que cumplen TODAS las reglas y elige la mejor según el OBJETIVO.
# Se usa búsqueda aleatoria y no un optimizador numérico porque el drawdown no es una función suave:
# así cada candidato se evalúa con el mismo backtest que se muestra al jurado.


def evaluar(W, R, universo, info_alta_vol, reglas_c):
    P = W @ R.T  # retorno diario de cada mezcla
    valor = np.cumprod(1 + P, axis=1)
    pico = np.maximum.accumulate(np.maximum(valor, 1.0), axis=1)
    m = {
        "ret_total": valor[:, -1] - 1,
        "vol": P.std(axis=1, ddof=1) * np.sqrt(252),
        "drawdown": (valor / pico - 1).min(axis=1),
    }
    m["ret_anual"] = (1 + m["ret_total"]) ** (252 / P.shape[1]) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        # Exceso sobre la CAJA expresada en la moneda del cliente (tasa USD + efecto cambiario),
        # para no comparar un retorno en COP/PEN contra una tasa en USD
        exceso = P - R[:, universo.index("CAJA")]
        m["sharpe"] = np.where(
            m["vol"] > 1e-9, exceso.mean(axis=1) * 252 / m["vol"], 0.0
        )
    acciones = [a for a in universo if not es_caja(a)]
    ia = [universo.index(a) for a in acciones]
    # Liquidez total = CAJA (USD) + CAJA_LOCAL. Para un cliente en USD la CAJA ya es su moneda local.
    local = W[:, universo.index("CAJA_LOCAL" if "CAJA_LOCAL" in universo else "CAJA")]
    caja = W[:, [universo.index(a) for a in universo if es_caja(a)]].sum(axis=1)

    def agrupado(col):
        grupos = sorted(INFO.loc[acciones, col].unique())
        M = np.array(
            [[INFO.at[a, col] == g for g in grupos] for a in acciones], dtype=float
        )
        return W[:, ia] @ M

    res = {}
    for r in reglas_c:
        t, p = r["rule_type"], r["params"]
        if t == "DD_MAX":
            res[r["clave"]] = (-m["drawdown"], p["limite"], "max")
        elif t == "VOL_MAX":
            res[r["clave"]] = (m["vol"], p["limite"], "max")
        elif t == "MAX_POR_SECTOR":
            res[r["clave"]] = (agrupado("sector").max(axis=1), p["limite"], "max")
        elif t == "MAX_POR_PAIS":
            res[r["clave"]] = (agrupado("country").max(axis=1), p["limite"], "max")
        elif t == "MAX_GRUPO":
            res[r["clave"]] = (W[:, ia] @ info_alta_vol, p["limite"], "max")
        elif t == "MIN_PAISES":
            res[r["clave"]] = (
                (agrupado("country") >= p["peso_min"] - 1e-9).sum(axis=1),
                p["n"],
                "min",
            )
        elif t == "MIN_SECTORES":
            res[r["clave"]] = (
                (agrupado("sector") >= p["peso_min"] - 1e-9).sum(axis=1),
                p["n"],
                "min",
            )
        elif t == "MAX_PESO_ACTIVO":
            res[r["clave"]] = (W[:, ia].max(axis=1), p["limite"], "max")
        elif t == "MIN_CAJA":
            res[r["clave"]] = (caja, p["limite"], "min")
        elif t == "MAX_CAJA":
            res[r["clave"]] = (caja, p["limite"], "max")
        elif t == "MIN_CAJA_LOCAL":
            res[r["clave"]] = (local, p["limite"], "min")
        elif t == "EXCLUIR_ACTIVO":
            res[r["clave"]] = (W[:, universo.index(p["asset_id"])], 0.0, "max")
    return m, res


def cumple_todo(res):
    ok = None
    for v, lim, sentido in res.values():
        c = v <= lim + 1e-9 if sentido == "max" else v >= lim - 1e-9
        ok = c if ok is None else ok & c
    return ok


def redondear(w, paso=PASO_PESO):
    # Pesos en múltiplos de 5% que suman 100% (método del mayor residuo)
    # Acepta un vector o una matriz (una mezcla por fila)
    W = np.atleast_2d(w)
    unidades = int(round(1 / paso))
    base_u = np.floor(W * unidades + 1e-9)
    faltan = (unidades - base_u.sum(axis=1)).astype(int)
    orden = np.argsort(-(W * unidades - base_u), axis=1)
    rango = np.argsort(
        orden, axis=1
    )  # posición de cada columna en el orden de residuos
    base_u += rango < faltan[:, None]
    out = base_u / unidades
    return out[0] if np.ndim(w) == 1 else out


def optimizar(
    cid, reglas_df, close_panel, umbral_alta_vol=UMBRAL_ALTA_VOL, n=N_CANDIDATOS
):
    rc = reglas_df[reglas_df["client_id"] == cid].to_dict("records")
    for i, r in enumerate(rc):
        r["clave"] = f"{r['rule_type']}#{i}"
    moneda = next(r["params"]["moneda"] for r in rc if r["rule_type"] == "MONEDA_BASE")
    objetivo = next(r["params"]["tipo"] for r in rc if r["rule_type"] == "OBJETIVO")
    restricciones = [r for r in rc if r["rule_type"] not in ("MONEDA_BASE", "OBJETIVO")]

    R_df = retornos_cliente(close_panel, moneda).dropna()
    universo = list(R_df.columns)
    R = R_df.values
    acciones = [a for a in universo if not es_caja(a)]
    vol_act = R_df[acciones].std() * np.sqrt(252)
    alta_vol = (vol_act > umbral_alta_vol).astype(float).values

    # Candidatos: la liquidez total se muestrea dentro de sus límites para no desperdiciar mezclas imposibles
    rng = np.random.default_rng(SEMILLA)
    min_local = max(
        [
            r["params"]["limite"]
            for r in restricciones
            if r["rule_type"] == "MIN_CAJA_LOCAL"
        ],
        default=0.0,
    )
    lo = max(
        [r["params"]["limite"] for r in restricciones if r["rule_type"] == "MIN_CAJA"]
        + [min_local],
        default=0.0,
    )
    hi = min(
        [r["params"]["limite"] for r in restricciones if r["rule_type"] == "MAX_CAJA"],
        default=1.0,
    )
    # Reglas contradictorias (mínimo > tope): se muestrea en el mínimo y la evaluación lo marcará como incumplido
    hi = max(hi, lo)
    caja = rng.uniform(lo, hi, n)
    resto = np.vstack(
        [
            rng.dirichlet(np.full(len(acciones), 0.5), n // 2),
            rng.dirichlet(np.ones(len(acciones)), n - n // 2)
            * (rng.random((n - n // 2, len(acciones))) < 0.6),
        ]
    )
    ia = [universo.index(a) for a in acciones]
    jc = universo.index("CAJA")
    jl = universo.index("CAJA_LOCAL") if "CAJA_LOCAL" in universo else None
    frac_local = rng.uniform(0, 1, n)

    def armar(caja, frac_local, resto):
        # Liquidez dentro de [lo, hi] repartida entre CAJA (USD) y CAJA_LOCAL (al menos min_local);
        # acciones en el resto; activos excluidos pasan a CAJA
        resto = np.where(resto.sum(axis=1, keepdims=True) == 0, 1.0, resto)
        resto = resto / resto.sum(axis=1, keepdims=True)
        W = np.zeros((len(caja), len(universo)))
        W[:, ia] = resto * (1 - caja)[:, None]
        if jl is None:
            W[:, jc] = caja
        else:
            loc = np.clip(np.maximum(frac_local * caja, min_local), 0, caja)
            W[:, jl], W[:, jc] = loc, caja - loc
        for r in restricciones:
            if r["rule_type"] == "EXCLUIR_ACTIVO":
                j = universo.index(r["params"]["asset_id"])
                W[:, jc] += W[:, j]
                W[:, j] = 0
        return W

    def violacion(res):
        # Cuánto incumple cada mezcla, sumando el exceso relativo sobre cada límite
        return (
            sum(
                np.maximum(0, (v - lim) if s == "max" else (lim - v))
                / max(abs(lim), 0.01)
                for v, lim, s in res.values()
            )
            if res
            else np.zeros(1)
        )

    def puntuar(m):
        return {
            "max_sharpe": m["sharpe"],
            "max_retorno": m["ret_anual"],
            "min_volatilidad": -m["vol"],
        }[objetivo]

    W = armar(caja, frac_local, resto)
    m, res = evaluar(W, R, universo, alta_vol, restricciones)

    # Refinamiento local: alrededor de las mejores mezclas (o de las que menos incumplen) se prueban
    # variaciones pequeñas. Encuentra soluciones que el muestreo puro deja pasar cuando las reglas son estrictas.
    for _ in range(2):
        fac = cumple_todo(res) if res else np.ones(len(W), bool)
        mejores = np.argsort(-np.where(fac, puntuar(m), -np.inf))[:40]
        cercanas = np.argsort(violacion(res) if res else np.zeros(len(W)))[:40]
        elites = W[np.unique(np.concatenate([mejores[fac[mejores]], cercanas]))]
        k = n // 4
        base_e = elites[rng.integers(len(elites), size=k)]
        t = rng.uniform(0.02, 0.25, k)[:, None]
        nuevo = (1 - t) * base_e + t * rng.dirichlet(np.full(len(universo), 0.5), k)
        liq = nuevo[:, jc] + (nuevo[:, jl] if jl is not None else 0)
        frac = nuevo[:, jl] / np.maximum(liq, 1e-12) if jl is not None else np.zeros(k)
        W = np.vstack([W, armar(np.clip(liq, lo, hi), frac, nuevo[:, ia])])
        m, res = evaluar(W, R, universo, alta_vol, restricciones)

    # La decisión se toma sobre pesos YA redondeados a 5%: lo que se recomienda es exactamente lo que se evalúa
    Wd = np.unique(redondear(W), axis=0)
    md, resd = evaluar(Wd, R, universo, alta_vol, restricciones)
    factible_d = cumple_todo(resd) if resd else np.ones(len(Wd), bool)
    if factible_d.any():
        # Desempate: entre las mezclas que empatan con la mejor (dentro de una tolerancia pequeña),
        # gana la de menor caída. A igual retorno, el cliente prefiere caer menos.
        punt = np.where(factible_d, puntuar(md), -np.inf)
        tol = {"max_retorno": 0.002, "max_sharpe": 0.01, "min_volatilidad": 0.001}[objetivo]
        empatadas = np.flatnonzero(punt >= punt.max() - tol)
        elegido, estado = Wd[empatadas[np.argmax(md["drawdown"][empatadas])]], "OK"
        # Ajuste fino: probar todos los movimientos de 5 puntos entre dos posiciones y aceptar el que
        # mejore el objetivo (o lo iguale con menor caída). Termina cuando ningún movimiento mejora:
        # así cada peso presentado es el mejor frente a sus variantes de 5 puntos.
        for _ in range(60):
            m0, _r = evaluar(elegido[None, :], R, universo, alta_vol, restricciones)
            p0, d0 = puntuar(m0)[0], m0["drawdown"][0]
            idx = [(i, j) for i in range(len(universo)) for j in range(len(universo))
                   if i != j and elegido[j] >= PASO_PESO - 1e-9]
            C = np.repeat(elegido[None, :], len(idx), axis=0)
            for f, (i, j) in enumerate(idx):
                C[f, i] += PASO_PESO
                C[f, j] -= PASO_PESO
            C = np.round(C, 10)
            mc, rcc = evaluar(C, R, universo, alta_vol, restricciones)
            ok = cumple_todo(rcc) if rcc else np.ones(len(C), bool)
            pc = puntuar(mc)
            mejor = ok & ((pc > p0 + tol) | ((pc >= p0) & (mc["drawdown"] > d0 + 1e-4)))
            if not mejor.any():
                break
            cand = np.flatnonzero(mejor)
            elegido = C[cand[np.lexsort((mc["drawdown"][cand], pc[cand]))[-1]]]
    else:
        # Ninguna mezcla cumple: se reporta la que menos incumple, para mostrar qué regla es imposible
        elegido, estado = Wd[np.argmin(violacion(resd))], "SIN_SOLUCION_FACTIBLE"
    factible = cumple_todo(res) if res else np.ones(len(W), bool)

    mw, rw = evaluar(elegido[None, :], R, universo, alta_vol, restricciones)
    return {
        "cid": cid,
        "moneda": moneda,
        "objetivo": objetivo,
        "estado": estado,
        "universo": universo,
        "pesos": elegido,
        "metricas": {k: float(v[0]) for k, v in mw.items()},
        "reglas": restricciones,
        "res": {k: (float(v[0]), lim, s) for k, (v, lim, s) in rw.items()},
        "R": R_df,
        "nube": (m, factible),
        "alta_vol": dict(zip(acciones, alta_vol.astype(bool))),
    }


resultados = {cid: optimizar(cid, reglas, close) for cid in clientes["client_id"]}
for cid, o in resultados.items():
    liquidez = sum(w for a, w in zip(o["universo"], o["pesos"]) if es_caja(a))
    if liquidez >= 0.9:
        # Señal de reglas mal traducidas (p. ej. sin MAX_CAJA y objetivo min_volatilidad): no es una recomendación útil
        print(
            f"ALERTA {cid}: {liquidez:.0%} en caja. Revisa gold_reglas_cliente (OBJETIVO y MAX_CAJA)."
        )

papel = met.set_index("asset_id")["papel"].to_dict()


def filas_recomendacion(cid, o, **extra):
    filas = []
    for a, w in zip(o["universo"], o["pesos"]):
        if w <= 0:
            continue
        nombre = {"CAJA": "Caja (depósito USD)", "CAJA_LOCAL": f"Caja en {o['moneda']}"}
        filas.append(
            {
                "client_id": cid,
                "asset_id": a,
                "peso": w,
                "nombre": nombre.get(a) or INFO.at[a, "name"],
                "sector": "Caja" if es_caja(a) else INFO.at[a, "sector"],
                "pais": "—" if es_caja(a) else INFO.at[a, "country"],
                "papel": "liquidez" if es_caja(a) else papel[a],
                "clase": clase(a),
                "moneda_cliente": o["moneda"],
                "objetivo": o["objetivo"],
                "estado": o["estado"],
                **{f"port_{k}": v for k, v in o["metricas"].items()},
                "fuente_precios": FUENTE,
                **extra,
                "run_id": RUN_ID,
            }
        )
    return filas


rec, cump, bt, nube = [], [], [], []
for cid, o in resultados.items():
    rec += filas_recomendacion(cid, o)
    for r in o["reglas"]:
        v, lim, s = o["res"][r["clave"]]
        holgura = (lim - v) if s == "max" else (v - lim)
        cump.append(
            {
                "client_id": cid,
                "regla": r["rule_type"],
                "params": json.dumps(r["params"], ensure_ascii=False),
                "origen": r["origen"],
                "supuesto": r["supuesto"],
                "evidencia_texto": r["evidencia_texto"],
                "valor_portafolio": v,
                "limite": lim,
                "sentido": s,
                "cumple": holgura >= -1e-9,
                "holgura": holgura,
                "regla_al_limite": abs(holgura) < 0.01,
                "run_id": RUN_ID,
            }
        )
    # Backtest: la mezcla elegida contra repartir igual en las 8 acciones, en la moneda del cliente
    R_df = o["R"]
    valor_p = (1 + R_df.values @ o["pesos"]).cumprod()
    acciones = [a for a in R_df.columns if not es_caja(a)]
    valor_ew = (1 + R_df[acciones].mean(axis=1)).cumprod()
    bt += [
        {"client_id": cid, "fecha": f, "valor_portafolio": vp, "valor_igual_peso": ve}
        for f, vp, ve in zip(R_df.index, valor_p, valor_ew)
    ]
    m, fac = o["nube"]
    idx = np.random.default_rng(SEMILLA).choice(
        len(fac), min(3000, len(fac)), replace=False
    )
    nube += [
        {
            "client_id": cid,
            "vol": m["vol"][i],
            "ret_anual": m["ret_anual"][i],
            "drawdown": m["drawdown"][i],
            "factible": bool(fac[i]),
            "elegido": False,
        }
        for i in idx
    ]
    nube.append(
        {
            "client_id": cid,
            "vol": o["metricas"]["vol"],
            "ret_anual": o["metricas"]["ret_anual"],
            "drawdown": o["metricas"]["drawdown"],
            "factible": o["estado"] == "OK",
            "elegido": True,
        }
    )

for nombre, filas in [
    ("gold_recomendacion", rec),
    ("gold_cumplimiento", cump),
    ("gold_backtest", bt),
    ("gold_frontera", nube),
]:
    guardar(pd.DataFrame(filas), nombre)
for cid, o in resultados.items():
    print(
        f"{cid} [{o['estado']}] {o['moneda']} {o['objetivo']}:",
        {a: f"{w:.0%}" for a, w in zip(o["universo"], o["pesos"]) if w > 0},
        "| ret",
        f"{o['metricas']['ret_total']:.1%}",
        "| dd",
        f"{o['metricas']['drawdown']:.1%}",
    )
display(pd.DataFrame(cump))

# COMMAND ----------

# ===== CELDA 7: SENSIBILIDAD: ¿qué supuesto cambia la recomendación? (gold_sensibilidad) =====
# Rotación = % del portafolio que habría que mover si el supuesto cambia (0% = misma recomendación).
otra = "diario" if FUENTE == "intradia" else "intradia"
try:
    close_otra = panel_precios(otra)[0].iloc[-(VENTANA + 1) :]
except Exception:
    close_otra = None

sens = []
for cid, o in resultados.items():
    base_w = dict(zip(o["universo"], o["pesos"]))
    dd = next(r for r in o["reglas"] if r["rule_type"] == "DD_MAX")["params"]["limite"]
    escenarios = []
    for delta in (-0.05, +0.05):
        rr = reglas.copy()
        msk = (rr["client_id"] == cid) & (rr["rule_type"] == "DD_MAX")
        rr.loc[msk, "params"] = [{"limite": round(dd + delta, 2)}]
        escenarios.append(
            (
                f"Tolerancia de caída {dd + delta:.0%} (hoy {dd:.0%})",
                rr,
                close,
                UMBRAL_ALTA_VOL,
            )
        )
    escenarios.append(
        (
            f"Umbral de alta volatilidad 20% (hoy {UMBRAL_ALTA_VOL:.0%})",
            reglas,
            close,
            0.20,
        )
    )
    if close_otra is not None:
        escenarios.append(
            (
                f"Precios del archivo {otra} (hoy {FUENTE})",
                reglas,
                close_otra,
                UMBRAL_ALTA_VOL,
            )
        )
    for nombre, rr, cp, umbral in escenarios:
        s = optimizar(cid, rr, cp, umbral, n=N_CANDIDATOS)
        nw = dict(zip(s["universo"], s["pesos"]))
        rot = 0.5 * sum(
            abs(nw.get(a, 0) - base_w.get(a, 0)) for a in set(nw) | set(base_w)
        )
        sens.append(
            {
                "client_id": cid,
                "escenario": nombre,
                "estado": s["estado"],
                "rotacion": rot,
                "pesos": json.dumps({a: round(w, 2) for a, w in nw.items() if w > 0}),
                "ret_total": s["metricas"]["ret_total"],
                "drawdown": s["metricas"]["drawdown"],
                "run_id": RUN_ID,
            }
        )
df_sens = pd.DataFrame(sens)
guardar(df_sens, "gold_sensibilidad")
display(df_sens.round(3))

# COMMAND ----------

# ===== CELDA 7B: TWIST: nueva información posterior a la fecha de corte (silver_twist) =====
# No trae precios nuevos: cada fila se convierte en (a) reglas nuevas si afecta a un cliente o
# (b) un escenario de estrés si es de mercado (tasas, tipo de cambio). Luego se compara antes vs. después.
HAY_TWIST = spark.catalog.tableExists(f"{DB}.silver_twist")
if not HAY_TWIST:
    print("Sin silver_twist: se omite esta celda.")
else:
    tw = leer("silver_twist")
    print("Twist vigente desde:", tw["fecha_efectiva"].max())

    # (a) Twists de cliente -> reglas. El agente propone, el código valida; si no hay agente, regla por texto.
    nuevas = []
    for t in tw[tw["tipo"] == "Client"].itertuples():
        cid = t.client_id
        actuales = reglas[reglas["client_id"] == cid][["rule_type", "params"]].to_dict(
            "records"
        )
        propuestas_tw = []
        if AGENTE_ACTIVO:
            prompt = f"""Llegó información nueva sobre el cliente {cid}. Decide qué reglas agregar o reemplazar.
Usa SOLO estos tipos: {catalogo_txt}
Reglas actuales del cliente: {json.dumps(actuales, ensure_ascii=False)}
Moneda base del cliente: {clientes.set_index("client_id").at[cid, "base_currency"]}
Si una regla nueva es del mismo tipo que una actual, la reemplaza.
Marca "supuesto": true si el número o la forma de cumplirla no está escrita literalmente en la noticia.
Responde solo una lista JSON: [{{"rule_type": "", "params": {{}}, "evidencia_texto": "", "supuesto": false}}]
NOTICIA ({t.release_id}): {t.new_information}
ACCIÓN REQUERIDA: {t.required_action}"""
            try:
                propuestas_tw = extraer_json(preguntar(prompt))
            except Exception as e:
                print(t.release_id, "respuesta no válida del agente:", e)
        validas = [
            p
            for p in propuestas_tw
            if validar(p.get("rule_type"), p.get("params"))[0]
            and p.get("rule_type") not in ("MONEDA_BASE", "DD_MAX")
        ]
        origen = "agente"
        if not validas and t.pct is not None and not np.isnan(t.pct):
            # Respaldo determinista: "necesitará disponer de X% del portafolio" -> liquidez mínima X%.
            # Que esa liquidez esté en la moneda del cliente es inferencia (la necesidad es en su moneda).
            origen = "regla_texto"
            validas = [
                {
                    "rule_type": "MIN_CAJA",
                    "params": {"limite": t.pct / 100},
                    "evidencia_texto": t.new_information,
                    "supuesto": False,
                },
                {
                    "rule_type": "MIN_CAJA_LOCAL",
                    "params": {"limite": t.pct / 100},
                    "evidencia_texto": "La necesidad de liquidez es en la moneda base del cliente",
                    "supuesto": True,
                },
            ]
        for p in validas:
            nuevas.append(
                {
                    "client_id": cid,
                    "rule_type": p["rule_type"],
                    "params": p["params"],
                    "origen": f"twist:{t.release_id}:{origen}",
                    "evidencia_texto": p.get("evidencia_texto", ""),
                    "supuesto": bool(p.get("supuesto", False)),
                }
            )

    # Colchón cambiario: si el twist también trae un shock de la moneda del cliente (TW_03: COP -8%), la
    # liquidez que retirará se protege contra su reversión: necesidad × (1 + shock), redondeado hacia arriba
    # a múltiplos de 5%. Ej. CL_02: 30% × 1.08 = 32.4% -> 35% (30% que retira + 5% de colchón).
    fx_shock = {str(t.field).split("_")[-1].upper(): t.pct / 100
                for t in tw[tw["tipo"] == "FX"].itertuples() if t.pct == t.pct}
    moneda_de = clientes.set_index("client_id")["base_currency"].to_dict()
    for n_ in nuevas:
        choque = fx_shock.get(moneda_de.get(n_["client_id"]))
        if n_["rule_type"] in ("MIN_CAJA", "MIN_CAJA_LOCAL") and choque and n_["origen"].startswith("twist:"):
            base_l = n_["params"]["limite"]
            con_colchon = min(1.0, np.ceil(round(base_l * (1 + choque) / PASO_PESO, 6)) * PASO_PESO)
            if con_colchon > base_l:
                n_["params"] = {"limite": round(con_colchon, 2)}
                n_["origen"] += ":colchon_fx"
                n_["supuesto"] = True
                n_["evidencia_texto"] = (f"{base_l:.0%} que retirará + colchón por shock cambiario de {choque:.0%}: "
                                         f"{base_l:.0%} × {1 + choque:.2f} = {base_l * (1 + choque):.1%} -> {con_colchon:.0%}")
                print(f"{n_['client_id']}: {n_['rule_type']} {base_l:.0%} -> {con_colchon:.0%} ({n_['evidencia_texto']})")

    reglas_twist = reglas.copy()
    for n_ in nuevas:
        reglas_twist = reglas_twist[
            ~(
                (reglas_twist["client_id"] == n_["client_id"])
                & (reglas_twist["rule_type"] == n_["rule_type"])
            )
        ]
        reglas_twist = pd.concat([reglas_twist, pd.DataFrame([n_])], ignore_index=True)
        # Si la nueva liquidez mínima supera el tope de caja vigente, el tope se sube (y queda registrado)
        if n_["rule_type"] in ("MIN_CAJA", "MIN_CAJA_LOCAL"):
            tope = (reglas_twist["client_id"] == n_["client_id"]) & (
                reglas_twist["rule_type"] == "MAX_CAJA"
            )
            if (
                tope.any()
                and reglas_twist.loc[tope, "params"].iloc[0]["limite"]
                < n_["params"]["limite"]
            ):
                reglas_twist.loc[tope, "params"] = [{"limite": n_["params"]["limite"]}]
                reglas_twist.loc[tope, "origen"] = f"{n_['origen']}:ajuste_tope"
    guardar(
        pd.DataFrame(nuevas).assign(
            params=lambda d: d["params"].map(json.dumps), run_id=RUN_ID
        )
        if nuevas
        else pd.DataFrame({"client_id": [""]}),
        "gold_reglas_twist",
    )
    display(pd.DataFrame(nuevas))

    # (b) Twists de mercado -> escenarios de estrés
    # Tasas: un alza de esta magnitud no tiene precedente en la ventana (no se puede estimar una beta),
    # así que se usa como aproximación el episodio de Stress observado (feb-2026), activo por activo.
    estres_hist = (
        close_hist.loc[: pd.Timestamp(ESTRES[1])].iloc[-1]
        / close_hist.loc[: pd.Timestamp(ESTRES[0]) - pd.Timedelta(days=1)].iloc[-1]
        - 1
    )
    escenarios = []
    for t in tw[tw["tipo"].isin(["Macro", "FX"])].itertuples():
        if t.tipo == "Macro":
            escenarios.append(
                {
                    "id": t.release_id,
                    "nombre": f"{t.release_id}: {t.new_information}",
                    # Acciones: réplica del estrés de feb-2026. Bono: pérdida por duración ante el alza de tasa.
                    "acciones": {
                        **estres_hist.to_dict(),
                        BONO: -DURACION_BONO * (t.pb if t.pb == t.pb else 0) / 10000,
                    },
                    "moneda": None,
                    "fx": 0.0,
                }
            )
        else:
            # "El COP se deprecia 8%": el riesgo hacia adelante es que se revierta (COP se aprecia 8%)
            moneda = t.field.split("_")[-1].upper()
            escenarios.append(
                {
                    "id": t.release_id,
                    "nombre": f"{t.release_id}: reversión de '{t.new_information}'",
                    "acciones": {},
                    "moneda": moneda,
                    "fx": (t.pct or 0) / 100,
                }
            )

    def impacto(pesos, universo, moneda_cli, escs):
        # Valor del portafolio tras aplicar uno o varios choques (1 = sin cambio)
        valor = 0.0
        for a, w in zip(universo, pesos):
            v = 1.0
            for e in escs:
                v *= 1 + e["acciones"].get(a, 0.0)
                if e["moneda"] == moneda_cli and a != "CAJA_LOCAL":
                    v /= 1 + e["fx"]
            valor += w * v
        return valor - 1

    # Todos los clientes: así cambiar de cliente en el dashboard no requiere volver a correr nada
    afectados = list(clientes["client_id"])
    rec_tw, estres = [], []
    resultados_twist = {}
    for cid in clientes["client_id"]:
        antes = resultados[cid]
        despues = optimizar(cid, reglas_twist, close) if cid in afectados else antes
        resultados_twist[cid] = despues
        if cid in afectados:
            rec_tw += filas_recomendacion(cid, antes, momento="antes")
            rec_tw += filas_recomendacion(cid, despues, momento="despues")
        dd_lim = next(
            r["params"]["limite"] for r in antes["reglas"] if r["rule_type"] == "DD_MAX"
        )
        aplica = [e for e in escenarios if e["moneda"] in (None, antes["moneda"])]
        combos = [[e] for e in aplica] + ([aplica] if len(aplica) > 1 else [])
        for momento, o in [("antes", antes), ("despues", despues)]:
            for esc in combos:
                imp = impacto(o["pesos"], o["universo"], o["moneda"], esc)
                estres.append(
                    {
                        "client_id": cid,
                        "momento": momento,
                        "escenario": " + ".join(e["id"] for e in esc),
                        "descripcion": " | ".join(e["nombre"] for e in esc),
                        "impacto": imp,
                        "limite_caida": dd_lim,
                        "cumple": -imp <= dd_lim + 1e-9,
                        "estado_motor": o["estado"],
                        "pesos": json.dumps(
                            {
                                a: round(w, 2)
                                for a, w in zip(o["universo"], o["pesos"])
                                if w > 0
                            }
                        ),
                        "run_id": RUN_ID,
                    }
                )
    guardar(pd.DataFrame(rec_tw), "gold_recomendacion_twist")
    if not estres:
        # El twist no trae noticias de tasas ni de tipo de cambio: se deja constancia en vez de una tabla vacía
        estres = [
            {
                "client_id": cid,
                "momento": "despues",
                "escenario": "sin escenarios de mercado",
                "descripcion": "",
                "impacto": 0.0,
                "limite_caida": 0.0,
                "cumple": True,
                "estado_motor": "",
                "pesos": "",
                "run_id": RUN_ID,
            }
            for cid in clientes["client_id"]
        ]
    guardar(pd.DataFrame(estres), "gold_estres_twist")
    display(
        pd.DataFrame(estres)[
            [
                "client_id",
                "momento",
                "escenario",
                "impacto",
                "limite_caida",
                "cumple",
                "pesos",
            ]
        ].round(3)
    )

# COMMAND ----------

# ===== CELDA 7C: MENÚ DE OPCIONES PARA EL CLIENTE (gold_opciones_cliente) =====
# Un asesor no presenta una sola cartera: presenta alternativas que cumplen TODAS las reglas del cliente
# (caída máxima, liquidez, twist) y explica qué gana y qué arriesga en cada una. El cliente elige cuánto
# riesgo adicional quiere tomar dentro de su límite.
OPCIONES = [
    (
        "Conservadora",
        "min_volatilidad",
        None,
        "La menor volatilidad posible dentro de las reglas",
    ),
    (
        "Balanceada",
        "max_sharpe",
        0.35,
        "El mejor retorno por unidad de riesgo, sin más de 35% en una empresa",
    ),
    (
        "Crecimiento",
        "max_retorno",
        None,
        "El mayor retorno que todavía respeta el límite de caída",
    ),
]
reglas_base = reglas_twist if HAY_TWIST else reglas
opciones, opciones_pesos = [], []
resultados_opciones = {}
for cid in clientes["client_id"]:
    for nombre_op, objetivo_op, tope_activo, idea in OPCIONES:
        rr = reglas_base[
            ~(
                (reglas_base["client_id"] == cid)
                & (reglas_base["rule_type"] == "OBJETIVO")
            )
        ]
        extra = [
            {
                "client_id": cid,
                "rule_type": "OBJETIVO",
                "params": {"tipo": objetivo_op},
                "origen": f"opcion:{nombre_op}",
                "evidencia_texto": idea,
                "supuesto": False,
            }
        ]
        if tope_activo:
            rr = rr[
                ~((rr["client_id"] == cid) & (rr["rule_type"] == "MAX_PESO_ACTIVO"))
            ]
            extra.append(
                {
                    "client_id": cid,
                    "rule_type": "MAX_PESO_ACTIVO",
                    "params": {"limite": tope_activo},
                    "origen": f"opcion:{nombre_op}",
                    "evidencia_texto": "Diversificación",
                    "supuesto": True,
                }
            )
        o = optimizar(
            cid, pd.concat([rr, pd.DataFrame(extra)], ignore_index=True), close
        )
        resultados_opciones[(cid, nombre_op)] = o
        pesos = {a: w for a, w in zip(o["universo"], o["pesos"]) if w > 0}
        fila = {
            "client_id": cid,
            "opcion": nombre_op,
            "objetivo": objetivo_op,
            "idea": idea,
            "estado": o["estado"],
            "n_empresas": sum(1 for a in pesos if clase(a) == "Renta variable"),
            "renta_fija": sum(w for a, w in pesos.items() if clase(a) == "Renta fija"),
            "renta_variable": sum(
                w for a, w in pesos.items() if clase(a) == "Renta variable"
            ),
            "liquidez": sum(w for a, w in pesos.items() if es_caja(a)),
            "ret_total": o["metricas"]["ret_total"],
            "drawdown": o["metricas"]["drawdown"],
            "vol": o["metricas"]["vol"],
            "pesos": json.dumps({a: round(w, 2) for a, w in pesos.items()}),
            "run_id": RUN_ID,
        }
        if HAY_TWIST:
            # Resistencia a las noticias del twist que aplican a la moneda del cliente, todas juntas
            aplica = [e for e in escenarios if e["moneda"] in (None, o["moneda"])]
            fila["estres_twist"] = (
                impacto(o["pesos"], o["universo"], o["moneda"], aplica)
                if aplica
                else 0.0
            )
            fila["resiste_twist"] = (
                -fila["estres_twist"]
                <= next(
                    r["params"]["limite"]
                    for r in o["reglas"]
                    if r["rule_type"] == "DD_MAX"
                )
                + 1e-9
            )
        opciones.append(fila)
        opciones_pesos += [
            {**f, "opcion": nombre_op} for f in filas_recomendacion(cid, o)
        ]
df_op = pd.DataFrame(opciones)
guardar(df_op, "gold_opciones_cliente")
guardar(pd.DataFrame(opciones_pesos), "gold_opciones_pesos")

# Asignación por clase de activo (liquidez / renta fija / renta variable) de cada cartera: la vista del asesor
carteras = [("recomendacion", rec)]
if HAY_TWIST:
    carteras.append(
        ("despues_twist", [f for f in rec_tw if f.get("momento") == "despues"])
    )
carteras += [
    (f"opcion:{op[0]}", [f for f in opciones_pesos if f["opcion"] == op[0]])
    for op in OPCIONES
]
df_clases = pd.concat(
    [
        pd.DataFrame(f)
        .groupby(["client_id", "clase"])["peso"]
        .sum()
        .reset_index()
        .assign(cartera=c)
        for c, f in carteras
        if f
    ],
    ignore_index=True,
)
guardar(df_clases, "gold_recomendacion_clases")
display(
    df_clases[df_clases["client_id"] == CLIENTE_ELEGIDO]
    .pivot_table(index="cartera", columns="clase", values="peso", fill_value=0)
    .round(2)
)
display(df_op[df_op["client_id"] == CLIENTE_ELEGIDO].drop(columns=["run_id"]).round(3))

# COMMAND ----------

# ===== CELDA 7D: ¿POR QUÉ ESE PORCENTAJE? (gold_justificacion_pesos) =====
# Para cada posición se calcula con datos: cuánto aporta al retorno y al riesgo, qué pasa si sube o baja
# 5 puntos (contra la caja, o contra la mayor acción si es caja) y qué regla fija ese peso.
# Así el porqué de "30% en liquidez" o "40% en Aurora Health" no depende del agente.
PASO_PRUEBA = 0.05


def incumplidas(res, reglas_c):
    nombres = {r["clave"]: f"{r['rule_type']} {json.dumps(r['params'], ensure_ascii=False)}" for r in reglas_c}
    return [nombres[k] for k, (v, lim, sent) in res.items()
            if (v[0] > lim + 1e-9 if sent == "max" else v[0] < lim - 1e-9)]


def justificar(cid, o, cartera):
    u, w, R = o["universo"], np.array(o["pesos"], dtype=float), o["R"].values
    acciones = [a for a in u if not es_caja(a)]
    av = np.array([float(o["alta_vol"][a]) for a in acciones])
    rc = o["reglas"]
    P = R @ w
    m0, _ = evaluar(w[None, :], R, u, av, rc)
    ret0 = m0["ret_total"][0]
    # Se compara con la misma medida que optimiza el motor para esta cartera
    medida = {"max_retorno": ("ret_anual", 1, 0.002, "retorno anual"), "max_sharpe": ("sharpe", 1, 0.01, "Sharpe"),
              "min_volatilidad": ("vol", -1, 0.001, "volatilidad")}[o["objetivo"]]
    obj0 = medida[1] * m0[medida[0]][0]
    dd_lim = next(r["params"]["limite"] for r in rc if r["rule_type"] == "DD_MAX")
    con_peso = [a for a in u if w[u.index(a)] > 0]
    filas_j = []
    for j, a in enumerate(u):
        if w[j] <= 0:
            continue
        # Contraparte: una inversión se compara contra la otra inversión de mayor peso (explica el reparto
        # entre ellas); la caja, contra la inversión de mayor peso (explica cuánto se invierte)
        otras = [x for x in con_peso if not es_caja(x) and x != a]
        cajas = [x for x in con_peso if es_caja(x)]
        contra = max(otras, key=lambda x: w[u.index(x)]) if otras else (cajas[0] if cajas else None)
        pruebas, rotas_dir, mejora, prudente = {}, {"sube": [], "baja": []}, False, []
        for nombre_p, d in (("sube", PASO_PRUEBA), ("baja", -PASO_PRUEBA)):
            w2 = w.copy()
            if contra is None:
                pruebas[nombre_p] = "no aplica"
                continue
            w2[j] += d
            w2[u.index(contra)] -= d
            if (w2 < -1e-9).any():
                pruebas[nombre_p] = "no aplica"
                continue
            m2, res2 = evaluar(w2[None, :], R, u, av, rc)
            rotas = incumplidas(res2, rc)
            rotas_dir[nombre_p] = rotas
            # Diferencias de retorno menores a 0.2 puntos se consideran empate
            d_obj = medida[1] * m2[medida[0]][0] - obj0
            d_ret, d_dd = m2["ret_total"][0] - ret0, m2["drawdown"][0] - m0["drawdown"][0]
            mejora |= (not rotas) and d_obj > medida[2]
            if not rotas and abs(d_obj) <= medida[2] and d_dd > 1e-4:
                prudente.append(f"con 5 puntos {'más' if d > 0 else 'menos'} el {medida[3]} casi no cambia "
                                f"(retorno {d_ret * 100:+.1f} pp) y la caída mejora a {m2['drawdown'][0]:.1%}")
            pruebas[nombre_p] = (f"retorno {m2['ret_total'][0]:.1%}, caída {m2['drawdown'][0]:.1%}"
                                 + (f"; incumple {', '.join(rotas)}" if rotas else "; cumple todas las reglas"))
        fijan = sorted(set(rotas_dir["sube"] + rotas_dir["baja"]))
        limites = []
        if rotas_dir["sube"]:
            limites.append(f"no puede subir 5 puntos sin incumplir {'; '.join(rotas_dir['sube'])}")
        if rotas_dir["baja"]:
            limites.append(f"no puede bajar 5 puntos sin incumplir {'; '.join(rotas_dir['baja'])}")
        ret_a = float((1 + R[:, j]).prod() - 1)
        aporte_riesgo = float(w[j] * np.cov(R[:, j], P)[0, 1] / P.var(ddof=1)) if P.var(ddof=1) > 0 else 0.0
        nombre = {"CAJA": "Caja (depósito USD)", "CAJA_LOCAL": f"Caja en {o['moneda']}"}.get(a) or INFO.at[a, "name"]
        if limites:
            motivo = "Peso acotado por las reglas: " + "; ".join(limites) + "."
        elif mejora:
            # Con pesos en múltiplos de 5% puede existir una variante algo mejor que la búsqueda no eligió
            motivo = f"Ninguna regla lo fija y una variante de 5 puntos mejora el {medida[3]}: el peso es aproximado."
        elif prudente:
            motivo = ("Ninguna regla lo fija; alternativa casi equivalente y más prudente: " + "; ".join(prudente)
                      + ". Útil si el cliente prefiere caer menos (ver opción Balanceada).")
        else:
            motivo = (f"Ninguna regla lo fija: moverlo 5 puntos empeora el {medida[3]}, así que es el mejor peso "
                      f"para el objetivo {o['objetivo']} dentro del límite de caída de {dd_lim:.0%}.")
        razon = (f"{nombre} {w[j]:.0%} ({clase(a)}). Rindió {ret_a:.1%} en el año en {o['moneda']}; aporta "
                 f"{w[j] * ret_a:+.1%} al retorno y {aporte_riesgo:.0%} del riesgo del portafolio. "
                 f"Con 5 puntos más (contra {contra}): {pruebas.get('sube', 'no aplica')}. "
                 f"Con 5 puntos menos: {pruebas.get('baja', 'no aplica')}. {motivo}")
        filas_j.append({"client_id": cid, "cartera": cartera, "asset_id": a, "nombre": nombre, "clase": clase(a),
                        "peso": w[j], "ret_activo": ret_a, "aporte_retorno": w[j] * ret_a, "aporte_riesgo": aporte_riesgo,
                        "contraparte": contra or "", "si_sube_5pp": pruebas.get("sube", "no aplica"),
                        "si_baja_5pp": pruebas.get("baja", "no aplica"), "reglas_que_lo_fijan": "; ".join(fijan),
                        "razon": razon, "run_id": RUN_ID})
    return filas_j


just = []
for cid in clientes["client_id"]:
    just += justificar(cid, resultados[cid], "recomendacion")
    if HAY_TWIST:
        just += justificar(cid, resultados_twist[cid], "despues_twist")
    for op in OPCIONES:
        just += justificar(cid, resultados_opciones[(cid, op[0])], f"opcion:{op[0]}")
df_just = pd.DataFrame(just)
guardar(df_just, "gold_justificacion_pesos")
cartera_final = "despues_twist" if HAY_TWIST else "recomendacion"
for r in df_just[(df_just["client_id"] == CLIENTE_ELEGIDO) & (df_just["cartera"] == cartera_final)].itertuples():
    print("•", r.razon)

# COMMAND ----------

# ===== CELDA 7E: CARTERAS PROPUESTAS POR EL EQUIPO (gold_carteras_equipo) =====
# Cualquier reparto que proponga alguien (widget carteras_equipo) se evalúa igual que el del motor:
# si cumple las reglas del cliente, retorno, caída, estrés del twist, el porqué de cada % y la diferencia
# con la recomendación. Así se discute con datos "35% en liquidez y 25% en Aurora" frente a la del motor.


def a_columna(clave, o):
    # Traduce lo que escribe una persona al código del universo del cliente
    k = str(clave).strip()
    alias = {"LIQUIDEZ": "CAJA_LOCAL" if "CAJA_LOCAL" in o["universo"] else "CAJA", "CAJA_USD": "CAJA",
             "CAJA": "CAJA", "CAJA_LOCAL": "CAJA_LOCAL", "BONO": BONO, BONO: BONO}
    col = alias.get(k.upper()) or (k if k in o["universo"] else None)
    if col is None:
        coincide = [a for a in o["universo"] if a in INFO.index and str(INFO.at[a, "name"]).lower() == k.lower()]
        col = coincide[0] if len(coincide) == 1 else None  # nombre ambiguo o desconocido: no se adivina
    return col if col in o["universo"] else None


try:
    propuestas_eq = json.loads(CARTERAS_EQUIPO) if CARTERAS_EQUIPO else {}
except json.JSONDecodeError as e:
    propuestas_eq = {}
    print("carteras_equipo no es un JSON válido:", e)

evaluadas, just_eq = [], []
for nombre_c, cfg in propuestas_eq.items():
    cid = cfg.get("cliente")
    if cid not in set(clientes["client_id"]):
        print(f"'{nombre_c}': cliente {cid} no existe; se omite.")
        continue
    ref = resultados_twist[cid] if HAY_TWIST else resultados[cid]  # reglas vigentes del cliente
    u = ref["universo"]
    w = np.zeros(len(u))
    no_reconocidos = []
    for clave, peso in cfg.get("pesos", {}).items():
        col = a_columna(clave, ref)
        if col is None:
            no_reconocidos.append(clave)
        else:
            w[u.index(col)] += float(peso)
    if abs(w.sum() - 100) < 1e-6:
        w = w / 100  # escrito en porcentaje (35, 25, ...): se pasa a fracción
        print(f"'{nombre_c}': pesos escritos en %, se convierten a fracción.")
    if (w < 0).any() or (w > 1).any() or not np.isfinite(w).all():
        print(f"'{nombre_c}': cada peso debe estar entre 0 y 1 (sin cortos ni apalancamiento); se omite.")
        continue
    if no_reconocidos or abs(w.sum() - 1) > 1e-6:
        print(f"'{nombre_c}': no reconocidos {no_reconocidos}, suma {w.sum():.0%}. Debe sumar 100% con nombres del universo.")
        continue
    acciones_c = [a for a in u if not es_caja(a)]
    av = np.array([float(ref["alta_vol"][a]) for a in acciones_c])
    m, res_c = evaluar(w[None, :], ref["R"].values, u, av, ref["reglas"])
    rotas = incumplidas(res_c, ref["reglas"])
    aplica = [e for e in escenarios if e["moneda"] in (None, ref["moneda"])] if HAY_TWIST else []
    estres_c = impacto(w, u, ref["moneda"], aplica) if aplica else np.nan
    dd_lim = next(r["params"]["limite"] for r in ref["reglas"] if r["rule_type"] == "DD_MAX")
    estres_ref = impacto(ref["pesos"], u, ref["moneda"], aplica) if aplica else np.nan
    veredicto = ("Cumple todas las reglas del cliente" if not rotas else f"Incumple: {'; '.join(rotas)}")
    if aplica and -estres_c > dd_lim:
        veredicto += f". No resiste el twist combinado ({estres_c:.1%} frente al límite de {dd_lim:.0%})"
    evaluadas.append({
        "client_id": cid, "cartera": nombre_c, "pesos": json.dumps({a: round(x, 2) for a, x in zip(u, w) if x > 0}),
        "liquidez": sum(x for a, x in zip(u, w) if es_caja(a)),
        "renta_fija": sum(x for a, x in zip(u, w) if clase(a) == "Renta fija"),
        "renta_variable": sum(x for a, x in zip(u, w) if clase(a) == "Renta variable"),
        "ret_total": m["ret_total"][0], "drawdown": m["drawdown"][0], "vol": m["vol"][0], "estres_twist": estres_c,
        "cumple": not rotas and not (aplica and -estres_c > dd_lim), "veredicto": veredicto,
        "ret_motor": ref["metricas"]["ret_total"], "drawdown_motor": ref["metricas"]["drawdown"], "estres_motor": estres_ref,
        "dif_retorno_vs_motor": m["ret_total"][0] - ref["metricas"]["ret_total"],
        "dif_caida_vs_motor": m["drawdown"][0] - ref["metricas"]["drawdown"], "run_id": RUN_ID,
    })
    just_eq += justificar(cid, {**ref, "pesos": w}, f"equipo:{nombre_c}")

if evaluadas:
    df_eq = pd.DataFrame(evaluadas)
    guardar(df_eq, "gold_carteras_equipo")
    df_just = pd.concat([df_just, pd.DataFrame(just_eq)], ignore_index=True)
    guardar(df_just, "gold_justificacion_pesos")
    for r in df_eq.itertuples():
        print(f"{r.cartera} [{r.client_id}] → {r.veredicto}. Retorno {r.ret_total:.1%} (motor {r.ret_motor:.1%}), "
              f"caída {r.drawdown:.1%} (motor {r.drawdown_motor:.1%}), twist {r.estres_twist:.1%} (motor {r.estres_motor:.1%}).")
        for j in just_eq:
            if j["cartera"] == f"equipo:{r.cartera}":
                print("   •", j["razon"])

# COMMAND ----------

# ===== CELDA 7F: SCORE DE RECOMENDACIÓN POR ACTIVO (gold_score_activos, gold_score_pesos) =====
# Score fijo de 0 a 100 por activo y por cliente, para el análisis del dashboard. Cada componente es el
# ranking del activo entre los 9 del universo (8 acciones + bono), medido en la moneda del cliente:
#   rentabilidad_riesgo: Sharpe (exceso sobre la caja en su moneda)  · control_caida: caída máxima
#   resistencia: estrés de feb-2026 + escenarios del twist           · momentum: retorno de 3 meses
#   fundamentales: crecimiento de utilidad, margen EBITDA, deuda neta/EBITDA · eventos: efecto propio a 5 días
# Las ponderaciones dependen del perfil y son fijas (quedan en gold_score_pesos). Sin dato (p. ej. fundamentales
# del bono), el componente no cuenta y se reparte su peso entre los demás.
PONDERACIONES = {
    "preservacion": {
        "rentabilidad_riesgo": 0.20,
        "control_caida": 0.25,
        "resistencia": 0.25,
        "momentum": 0.10,
        "fundamentales": 0.10,
        "eventos": 0.10,
    },
    "crecimiento": {
        "rentabilidad_riesgo": 0.25,
        "control_caida": 0.10,
        "resistencia": 0.10,
        "momentum": 0.25,
        "fundamentales": 0.15,
        "eventos": 0.15,
    },
    "balanceado": {
        "rentabilidad_riesgo": 0.25,
        "control_caida": 0.15,
        "resistencia": 0.15,
        "momentum": 0.15,
        "fundamentales": 0.15,
        "eventos": 0.15,
    },
}
UMBRAL_SI = (
    60  # score mínimo para "Sí · alternativa" cuando el activo no está en la cartera
)


def perfil_score(cid):
    dd = next(
        r["params"]["limite"]
        for r in resultados[cid]["reglas"]
        if r["rule_type"] == "DD_MAX"
    )
    if dd <= 0.10:
        return "preservacion"
    return (
        "crecimiento" if resultados[cid]["objetivo"] == "max_retorno" else "balanceado"
    )


def rango(serie, mayor_es_mejor=True):
    # Posición 0..1 entre los activos con dato (1 = el mejor)
    r = serie.rank(ascending=mayor_es_mejor, method="average")
    n = serie.notna().sum()
    return (r - 1) / (n - 1) if n > 1 else r * 0 + 0.5


def dato(a, c, v, con_twist):
    # Valor real detrás de cada componente, para la razón en texto
    if c == "rentabilidad_riesgo":
        return f"Sharpe {v.at[a, 'sharpe']:.2f}"
    if c == "control_caida":
        return f"caída máxima {v.at[a, 'max_drawdown']:.1%}"
    if c == "resistencia":
        return f"estrés feb {v.at[a, 'estres_feb']:.1%}" + (
            f", twist {v.at[a, 'estres_twist']:.1%}" if con_twist else ""
        )
    if c == "momentum":
        return f"retorno 3 meses {v.at[a, 'retorno_3m']:.1%}"
    if c == "fundamentales":
        if pd.isna(v.at[a, "crec_utilidad_12m"]):
            return "sin fundamentales (no es empresa)"
        return f"utilidad {v.at[a, 'crec_utilidad_12m']:+.1%}, deuda neta/EBITDA {v.at[a, 'deuda_neta_ebitda']:.2f}"
    if v.at[a, "peor_evento_5d"] != 0:
        return f"evento con efecto propio {v.at[a, 'peor_evento_5d']:+.1%} a 5 días"
    return "sin eventos negativos"


eventos_df = leer("gold_impacto_eventos")
peor_evento = eventos_df.groupby("asset_id")["efecto_propio_5d"].min()
met_i = met.set_index("asset_id")
scores = []
for cid in clientes["client_id"]:
    o = resultados_twist[cid] if HAY_TWIST else resultados[cid]
    R_df = o["R"]
    activos_s = [a for a in R_df.columns if not es_caja(a)]
    caja_cli = R_df["CAJA"]
    aplica = (
        [e for e in escenarios if e["moneda"] in (None, o["moneda"])]
        if HAY_TWIST
        else []
    )
    v = pd.DataFrame(index=activos_s)
    v["sharpe"] = [
        (R_df[a] - caja_cli).mean() * 252 / (R_df[a].std() * np.sqrt(252))
        for a in activos_s
    ]
    v["max_drawdown"] = [max_drawdown(R_df[a]) for a in activos_s]
    v["retorno_12m"] = [(1 + R_df[a]).prod() - 1 for a in activos_s]
    v["estres_feb"] = [
        (1 + R_df[a].loc[pd.Timestamp(ESTRES[0]) : pd.Timestamp(ESTRES[1])]).prod() - 1
        for a in activos_s
    ]
    v["estres_twist"] = [
        impacto([1.0], [a], o["moneda"], aplica) if aplica else np.nan
        for a in activos_s
    ]
    v["retorno_3m"] = met_i.loc[activos_s, "retorno_3m"].astype(float)
    v["crec_utilidad_12m"] = met_i.loc[activos_s, "crec_utilidad_12m"].astype(float)
    v["margen_ebitda"] = met_i.loc[activos_s, "margen_ebitda"].astype(float)
    v["deuda_neta_ebitda"] = met_i.loc[activos_s, "deuda_neta_ebitda"].astype(float)
    v["peor_evento_5d"] = peor_evento.reindex(activos_s).fillna(
        0.0
    )  # sin evento = 0 (neutral)

    comp = pd.DataFrame(index=activos_s)
    comp["rentabilidad_riesgo"] = rango(v["sharpe"])
    comp["control_caida"] = rango(v["max_drawdown"])  # menos negativa = mejor
    comp["resistencia"] = pd.concat(
        [rango(v["estres_feb"]), rango(v["estres_twist"])], axis=1
    ).mean(axis=1)
    comp["momentum"] = rango(v["retorno_3m"])
    comp["fundamentales"] = pd.concat(
        [
            rango(v["crec_utilidad_12m"]),
            rango(v["margen_ebitda"]),
            rango(v["deuda_neta_ebitda"], mayor_es_mejor=False),
        ],
        axis=1,
    ).mean(axis=1)
    comp["eventos"] = rango(v["peor_evento_5d"])

    perfil = perfil_score(cid)
    pond = pd.Series(PONDERACIONES[perfil])
    peso_util = comp.notna().mul(pond, axis=1).sum(axis=1)
    score = (comp.fillna(0).mul(pond, axis=1).sum(axis=1) / peso_util * 100).round(1)
    ranking = score.rank(ascending=False, method="min").astype(int)
    pesos_cartera = dict(zip(o["universo"], o["pesos"]))

    for a in activos_s:
        c_ok = comp.loc[a].dropna()
        aportes = (c_ok - 0.5) * pond[
            c_ok.index
        ]  # cuánto empuja cada componente sobre/bajo la media
        fuertes = [
            c for c in aportes.sort_values(ascending=False).index[:2] if aportes[c] > 0
        ]
        debiles = [c for c in aportes.sort_values().index[:2] if aportes[c] < 0]
        peso = float(pesos_cartera.get(a, 0.0))
        if peso > 0:
            veredicto = f"Sí · en la cartera ({peso:.0%})"
        elif score[a] >= UMBRAL_SI:
            veredicto = "Sí · alternativa"
        else:
            veredicto = "No"
        razon = (
            f"Score {score[a]:.0f}/100 (puesto {ranking[a]} de {len(activos_s)}, perfil {perfil}). "
            f"A favor: {'; '.join(dato(a, c, v, bool(aplica)) for c in fuertes) or 'ningún componente sobre la media'}. "
            f"En contra: {'; '.join(dato(a, c, v, bool(aplica)) for c in debiles) or 'ningún componente bajo la media'}."
        )
        # Dónde aparece el activo en las 3 opciones: explica un "Sí · alternativa" con datos
        en_opciones = [f"{op[0]} {resultados_opciones[(cid, op[0])]['pesos'][resultados_opciones[(cid, op[0])]['universo'].index(a)]:.0%}"
                       for op in OPCIONES
                       if a in resultados_opciones[(cid, op[0])]["universo"]
                       and resultados_opciones[(cid, op[0])]["pesos"][resultados_opciones[(cid, op[0])]["universo"].index(a)] > 0]
        if peso == 0 and score[a] >= UMBRAL_SI:
            razon += (f" No entra en la cartera principal porque su objetivo ({o['objetivo']}) prioriza otros activos"
                      + (f"; sí aparece en: {', '.join(en_opciones)}." if en_opciones else "."))
        elif peso == 0 and score[a] >= UMBRAL_SI - 10:
            razon += " Cerca del umbral: candidato si cambian las condiciones."
        elif peso > 0 and score[a] < UMBRAL_SI:
            razon += (f" Está en la cartera por su aporte al objetivo ({o['objetivo']}); su score bajo marca el riesgo a "
                      f"vigilar: {'; '.join(dato(a, c, v, bool(aplica)) for c in debiles) or 'ver componentes'}.")
        scores.append(
            {
                "client_id": cid,
                "asset_id": a,
                "nombre": INFO.at[a, "name"],
                "clase": clase(a),
                "score": float(score[a]),
                "ranking": int(ranking[a]),
                "veredicto": veredicto,
                "peso_en_cartera": peso,
                "perfil_score": perfil,
                **{
                    f"comp_{c}": (
                        round(float(comp.at[a, c]) * 100, 1)
                        if pd.notna(comp.at[a, c])
                        else np.nan
                    )
                    for c in comp.columns
                },
                **{
                    k: (float(v.at[a, k]) if pd.notna(v.at[a, k]) else np.nan)
                    for k in v.columns
                },
                "razon": razon,
                "run_id": RUN_ID,
            }
        )

df_score = pd.DataFrame(scores).sort_values(["client_id", "ranking"])
guardar(df_score, "gold_score_activos")
guardar(
    pd.DataFrame(
        [
            {"perfil": p_, "componente": c, "ponderacion": w_}
            for p_, d_ in PONDERACIONES.items()
            for c, w_ in d_.items()
        ]
    ),
    "gold_score_pesos",
)
display(
    df_score[df_score["client_id"] == CLIENTE_ELEGIDO][
        ["ranking", "nombre", "score", "veredicto", "razon"]
    ]
)

# COMMAND ----------
# ===== CELDA 8: AGENTE EXPLICADOR (brief del cliente elegido, citando tablas Gold) =====
# Mismo ciclo que el agente investigador: Gemini pide una herramienta, el código la ejecuta.
# Las secciones responden las 5 preguntas del Decision Dashboard.
MAX_PASOS = 18  # 11 consultas obligatorias + margen para una entrega rechazada
PCT = [
    "impacto",
    "ret_motor",
    "drawdown_motor",
    "aporte_retorno",
    "aporte_riesgo",
    "renta_fija",
    "renta_variable",
    "liquidez",
    "estres_twist",
    "vol",
    "limite_caida",
    "peso",
    "retorno_12m",
    "retorno_3m",
    "vol_anual",
    "max_drawdown",
    "retorno_estres",
    "rotacion",
    "ret_total",
    "drawdown",
    "valor_portafolio",
    "limite",
    "holgura",
    "ret_5d",
    "efecto_propio_5d",
    "ret_hasta_corte",
    "port_ret_total",
    "port_drawdown",
    "port_vol",
    "margen_ebitda",
    "crec_ventas_12m",
    "crec_utilidad_12m",
]


def a_texto(df):
    # Fracciones como % con 1 decimal: así el agente cita "10.0%" igual que la tabla
    # MIN_PAISES / MIN_SECTORES son conteos, no fracciones: esas filas no se pasan a %
    d = df.copy()
    conteo = (
        d["regla"].isin(["MIN_PAISES", "MIN_SECTORES"])
        if "regla" in d
        else pd.Series(False, index=d.index)
    )
    for c in d.columns:
        if c in PCT and pd.api.types.is_numeric_dtype(d[c]):
            # Vacíos (p. ej. margen EBITDA del bono, que no es una empresa) se dejan como "n/a"
            pct = d[c].map(lambda v: "n/a" if pd.isna(v) else f"{v * 100:.1f}%")
            entero = d[c].map(lambda v: "n/a" if pd.isna(v) else str(int(round(v))))
            d[c] = np.where(conteo, entero, pct)
    return d.astype(str).to_json(orient="records", force_ascii=False)


HERRAMIENTAS = {
    "ver_recomendacion": lambda: a_texto(
        leer("gold_recomendacion").query("client_id == @CLIENTE_ELEGIDO")[
            [
                "asset_id",
                "nombre",
                "sector",
                "pais",
                "papel",
                "clase",
                "peso",
                "estado",
                "objetivo",
                "moneda_cliente",
                "port_ret_total",
                "port_drawdown",
                "port_vol",
            ]
        ]
    ),
    "ver_cumplimiento": lambda: a_texto(
        leer("gold_cumplimiento").query("client_id == @CLIENTE_ELEGIDO")[
            [
                "regla",
                "params",
                "origen",
                "supuesto",
                "evidencia_texto",
                "valor_portafolio",
                "limite",
                "cumple",
                "regla_al_limite",
            ]
        ]
    ),
    "ver_metricas": lambda: a_texto(
        leer("gold_metricas_activo")[
            [
                "asset_id",
                "name",
                "sector",
                "country",
                "retorno_12m",
                "retorno_3m",
                "vol_anual",
                "max_drawdown",
                "retorno_estres",
                "papel",
                "clase",
                "margen_ebitda",
                "deuda_neta_ebitda",
            ]
        ]
    ),
    "ver_eventos": lambda: a_texto(leer("gold_impacto_eventos")),
    "ver_sensibilidad": lambda: a_texto(
        leer("gold_sensibilidad").query("client_id == @CLIENTE_ELEGIDO")[
            ["escenario", "estado", "rotacion", "pesos"]
        ]
    ),
    "ver_calidad_datos": lambda: (
        leer("silver_bitacora_calidad")
        .groupby("accion")["filas_afectadas"]
        .sum()
        .to_json()
        + " | "
        + leer("silver_conciliacion_fuentes").round(2).to_json(orient="records")
    ),
    "ver_cliente": lambda: (
        leer("silver_clientes")
        .query("client_id == @CLIENTE_ELEGIDO")
        .astype(str)
        .to_json(orient="records", force_ascii=False)
    ),
}
OBLIGATORIAS = {
    "ver_cliente",
    "ver_recomendacion",
    "ver_cumplimiento",
    "ver_sensibilidad",
    "ver_metricas",
    "ver_eventos",
}
if HAY_TWIST:
    # Con información nueva, el brief debe explicar qué cambió y si la recomendación resiste los escenarios
    HERRAMIENTAS["ver_twist"] = lambda: (
        leer("silver_twist").astype(str).to_json(orient="records", force_ascii=False)
        + " | "
        + a_texto(
            leer("gold_recomendacion_twist").query("client_id == @CLIENTE_ELEGIDO")[
                ["momento", "asset_id", "nombre", "peso", "estado"]
            ]
        )
        + " | "
        + a_texto(
            leer("gold_estres_twist").query("client_id == @CLIENTE_ELEGIDO")[
                ["momento", "escenario", "impacto", "limite_caida", "cumple"]
            ]
        )
    )
    OBLIGATORIAS.add("ver_twist")
    # La recomendación vigente es la posterior al twist: ver_recomendacion la devuelve a ella
    HERRAMIENTAS["ver_recomendacion"] = lambda: a_texto(
        leer("gold_recomendacion_twist").query(
            "client_id == @CLIENTE_ELEGIDO and momento == 'despues'"
        )[
            [
                "asset_id",
                "nombre",
                "sector",
                "pais",
                "papel",
                "clase",
                "peso",
                "estado",
                "objetivo",
                "moneda_cliente",
                "port_ret_total",
                "port_drawdown",
                "port_vol",
            ]
        ]
    )

# Menú de opciones: el brief las presenta para que el cliente elija cuánto riesgo adicional tomar
HERRAMIENTAS["ver_opciones"] = lambda: a_texto(
    leer("gold_opciones_cliente").query("client_id == @CLIENTE_ELEGIDO")[
        [
            c
            for c in [
                "opcion",
                "idea",
                "estado",
                "n_empresas",
                "liquidez",
                "ret_total",
                "drawdown",
                "estres_twist",
                "resiste_twist",
                "pesos",
            ]
            if c in leer("gold_opciones_cliente").columns
        ]
    ]
)
OBLIGATORIAS.add("ver_opciones")

# El porqué de cada porcentaje, calculado por código (celda 7D)
HERRAMIENTAS["ver_justificacion"] = lambda: a_texto(
    leer("gold_justificacion_pesos").query(
        "client_id == @CLIENTE_ELEGIDO and cartera == @cartera_final")[
        ["asset_id", "nombre", "clase", "peso", "aporte_retorno", "aporte_riesgo", "razon"]]
)
OBLIGATORIAS.add("ver_justificacion")
# Score fijo por activo: el brief lo usa para explicar por qué sí o por qué no cada activo
HERRAMIENTAS["ver_score"] = lambda: a_texto(
    leer("gold_score_activos").query("client_id == @CLIENTE_ELEGIDO")[
        ["ranking", "asset_id", "nombre", "score", "veredicto", "razon"]])
OBLIGATORIAS.add("ver_score")
if spark.catalog.tableExists(f"{DB}.gold_carteras_equipo") and evaluadas:
    # Carteras del equipo para este cliente: el brief las compara con la del motor
    HERRAMIENTAS["ver_carteras_equipo"] = lambda: a_texto(
        leer("gold_carteras_equipo").query("client_id == @CLIENTE_ELEGIDO")[
            ["cartera", "pesos", "ret_total", "drawdown", "estres_twist", "veredicto", "ret_motor", "drawdown_motor"]])


def instrucciones_para(CLIENTE_ELEGIDO):
    return f"""Eres analista de inversiones. Prepara el brief de la recomendación para el cliente {CLIENTE_ELEGIDO}.
Fecha de decisión: {FECHA_CORTE.date()}. Fuente de precios usada: {FUENTE}.
Herramientas (una por paso, sin argumentos): {sorted(HERRAMIENTAS)}.
Antes de entregar es OBLIGATORIO usar: {sorted(OBLIGATORIAS)}.
Reglas:
- Toda cifra debe copiarse tal cual de una herramienta y citar la tabla de donde sale.
- No propongas pesos distintos a los de ver_recomendacion: el motor ya decidió.
- "recomendacion": el PRIMER punto resume la asignación por clase (Liquidez, Renta fija, Renta variable) con sus %
  y por qué esa proporción (límite de caída, liquidez requerida, efecto de tasas en la renta fija). Luego UN punto por cada posición con peso > 0, con nombre de la empresa (o tipo de caja), sector, país,
  peso, y el porqué de ESE porcentaje tomado de ver_justificacion (razon: aporte al retorno y al riesgo, qué pasa
  con 5 puntos más o menos y qué regla lo fija), más una métrica de ver_metricas.
- "senales": qué activos quedaron fuera y por qué, con el score y la razon de ver_score (veredicto No), y qué
  evento explica movimientos.
- "significado_cliente": qué implica la recomendación para este cliente (moneda, liquidez, límite de caída).
- En "recomendacion", después de las posiciones, agrega UN punto por cada opción de ver_opciones (Conservadora,
  Balanceada, Crecimiento) con su retorno, caída y si resiste el twist, para que el cliente elija cuánto riesgo tomar.
- En "tabla" escribe el nombre de la herramienta que usaste (ej. ver_metricas).
- Si existe ver_carteras_equipo y trae carteras de este cliente: en "senales" compara cada una con la del motor
  (retorno, caída, twist y veredicto) y di por qué se recomienda una u otra.
- Si estado = SIN_SOLUCION_FACTIBLE, dilo y explica qué regla no se puede cumplir.
- "riesgos_supuestos" debe incluir la fuente de precios y cualquier regla con supuesto = true o al límite.
- Si existe ver_twist: la recomendación final es la de momento = despues; explica qué noticia la cambió
  y si resiste cada escenario (cumple = True/False) contra su límite de caída.
Responde SIEMPRE solo un JSON con una de estas formas:
{{"accion": "<herramienta>"}}
{{"accion": "entregar", "brief": {{
  "que_ocurre": [{{"texto": "", "tabla": ""}}],
  "senales": [{{"texto": "", "tabla": ""}}],
  "significado_cliente": [{{"texto": "", "tabla": ""}}],
  "recomendacion": [{{"texto": "", "tabla": ""}}],
  "riesgos_supuestos": [{{"texto": "", "tabla": ""}}]}}}}"""


TABLA_DE = {
    "ver_recomendacion": {
        "gold_recomendacion_twist" if HAY_TWIST else "gold_recomendacion"
    },
    "ver_cumplimiento": {"gold_cumplimiento"},
    "ver_metricas": {"gold_metricas_activo"},
    "ver_eventos": {"gold_impacto_eventos"},
    "ver_sensibilidad": {"gold_sensibilidad"},
    "ver_calidad_datos": {"silver_bitacora_calidad", "silver_conciliacion_fuentes"},
    "ver_cliente": {"silver_clientes"},
    "ver_twist": {"silver_twist", "gold_recomendacion_twist", "gold_estres_twist"},
    "ver_opciones": {"gold_opciones_cliente"},
    "ver_justificacion": {"gold_justificacion_pesos"},
    "ver_carteras_equipo": {"gold_carteras_equipo"},
    "ver_score": {"gold_score_activos"},
}

# Escalable: un brief por cliente (widget brief_clientes = "todos" o una lista "CL_01,CL_03").
# HERRAMIENTAS filtra por la variable CLIENTE_ELEGIDO, que aquí toma el valor de cada cliente.
CLIENTES_BRIEF = (
    list(clientes["client_id"])
    if BRIEF_CLIENTES == "todos"
    else [c.strip() for c in BRIEF_CLIENTES.split(",") if c.strip()]
)


def alias_de(r):
    # Patrones que cuentan como "mencionar" una posición en el brief
    if r.asset_id == "CAJA":
        return [r"CAJA(?!_)", r"dep[oó]sito (en )?USD", r"caja (en )?(USD|d[oó]lares)"]
    if r.asset_id == "CAJA_LOCAL":
        return [
            r"CAJA_LOCAL",
            rf"caja (en )?{r.moneda_cliente}",
            r"caja local",
            r"moneda local",
        ]
    return [
        rf"{re.escape(r.asset_id)}",
        re.escape(r.nombre),
        rf"{re.escape(r.nombre.split()[0])}",
    ]


briefs, trazas = [], []
if not AGENTE_ACTIVO:
    print("Agente desactivado, no se genera el brief. Motivo:", MOTIVO_AGENTE)
for CLIENTE_ELEGIDO in CLIENTES_BRIEF if AGENTE_ACTIVO else []:
    print(f"--- Brief {CLIENTE_ELEGIDO}")
    historial, traza, brief = "", [], None
    for paso in range(1, MAX_PASOS + 1):
        try:
            d = extraer_json(
                preguntar(
                    instrucciones_para(CLIENTE_ELEGIDO)
                    + "\n\nHISTORIAL:\n"
                    + (historial or "Ninguno.")
                ),
                "{",
                "}",
            )
        except Exception:
            historial += f"\nPaso {paso}: respuesta no válida, responde solo JSON."
            continue
        accion = d.get("accion")
        if accion == "entregar":
            usadas = {t["accion"] for t in traza}
            # Validación por código: la sección recomendacion debe nombrar cada posición del motor
            rec_final = (
                leer("gold_recomendacion_twist").query(
                    "client_id == @CLIENTE_ELEGIDO and momento == 'despues'"
                )
                if HAY_TWIST
                else leer("gold_recomendacion").query("client_id == @CLIENTE_ELEGIDO")
            )
            texto_rec = " ".join(
                i.get("texto", "")
                for i in (d.get("brief") or {}).get("recomendacion", [])
            )
            sin_mencion = [
                r.nombre
                for r in rec_final.itertuples()
                # Alias por posición: código (como palabra completa), nombre o su primera palabra.
                # Las dos cajas tienen alias propios para que "Caja" no cubra a ambas.
                if not any(re.search(patron, texto_rec, re.I) for patron in alias_de(r))
            ]
            if OBLIGATORIAS <= usadas and sin_mencion:
                historial += (
                    f"\nPaso {paso}: entrega rechazada, en 'recomendacion' falta un punto para: {sin_mencion} "
                    "(nombre, peso y porqué con una métrica)."
                )
                traza.append(
                    {
                        "paso": paso,
                        "accion": "entregar",
                        "resultado": f"rechazado: no menciona {sin_mencion}",
                    }
                )
                continue
            if not OBLIGATORIAS <= usadas:
                falta = sorted(OBLIGATORIAS - usadas)
                historial += f"\nPaso {paso}: entrega rechazada, primero usa {falta}."
                traza.append(
                    {
                        "paso": paso,
                        "accion": "entregar",
                        "resultado": f"rechazado: falta {falta}",
                    }
                )
                continue
            brief = d.get("brief", {})
            traza.append(
                {"paso": paso, "accion": "entregar", "resultado": "brief entregado"}
            )
            break
        resultado = (
            HERRAMIENTAS[accion]()
            if accion in HERRAMIENTAS
            else "Herramienta desconocida."
        )
        historial += f"\nPaso {paso}: {accion} -> {resultado}"
        traza.append({"paso": paso, "accion": accion, "resultado": resultado[:500]})
        print(f"Paso {paso}: {accion}")

    # Validación por código: la tabla citada debe haberse consultado y cada cifra debe aparecer en lo consultado
    consultadas = set().union(
        *[TABLA_DE[t["accion"]] for t in traza if t["accion"] in TABLA_DE]
    )
    cifras_fuente = set(re.findall(r"-?\d+(?:\.\d+)?", historial))
    filas = []
    for seccion, items in (brief or {}).items():
        for it in items if isinstance(items, list) else []:
            cifras = re.findall(r"-?\d+(?:\.\d+)?", it.get("texto", ""))
            sin_respaldo = [x for x in cifras if x not in cifras_fuente]
            filas.append(
                {
                    "client_id": CLIENTE_ELEGIDO,
                    "seccion": seccion,
                    "texto": it.get("texto", ""),
                    "tabla": it.get("tabla", ""),
                    # Se acepta el nombre de la tabla o el de la herramienta que la consulta
                    "tabla_consultada": bool(
                        TABLA_DE.get(it.get("tabla", ""), {it.get("tabla", "")})
                        & consultadas
                    ),
                    "cifras_sin_respaldo": ",".join(sin_respaldo),
                    "run_id": RUN_ID,
                }
            )
    if brief is None:
        print(CLIENTE_ELEGIDO, "· el agente no entregó brief en", MAX_PASOS, "pasos.")
    briefs += filas
    trazas += [{**t, "client_id": CLIENTE_ELEGIDO, "run_id": RUN_ID} for t in traza]

if briefs:
    guardar(pd.DataFrame(briefs), "gold_brief_cliente")
    display(pd.DataFrame(briefs))
if trazas:
    guardar(pd.DataFrame(trazas), "gold_traza_agente")
