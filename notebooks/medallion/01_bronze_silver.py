# Databricks notebook source
# MAGIC %md
# MAGIC # Value Finders · 01 · Bronze → Silver
# MAGIC **Pregunta:** ¿Qué decisión de inversión recomendarían a este cliente hoy, a partir de la información disponible, y qué evidencia les permite defenderla?
# MAGIC
# MAGIC | CRISP-DM | Capa | Celda |
# MAGIC |---|---|---|
# MAGIC | 2. Entendimiento de los datos | Bronze | 1 · Lectura tal cual + perfil |
# MAGIC | 3. Preparación de los datos | Silver | 2–7 · Limpieza con regla, acción y porqué |
# MAGIC | 3. Preparación (verificación) | Silver | 8 · Conciliación de fuentes de precio + bitácora |
# MAGIC
# MAGIC **Principio de limpieza:** nada se borra sin motivo. Cada regla deja una de estas acciones en `silver_bitacora_calidad`:
# MAGIC - `CORREGIR`: el valor está mal escrito pero el dato es recuperable con certeza (formato, código, escala).
# MAGIC - `MARCAR`: el dato falta o es dudoso; la fila se queda y se marca para que el modelador decida.
# MAGIC - `EXCLUIR_ANALISIS`: la fila se queda en Silver, pero no sirve para calcular retornos (por ejemplo, vela sin cierre).
# MAGIC - `ELIMINAR_DUPLICADO`: la misma fila aparece dos veces; contarla dos veces sesgaría los retornos. Se conserva una.

# COMMAND ----------

# ===== CELDA 0: CONFIGURACIÓN =====
import hashlib, json, re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Parámetros: el job del bundle (resources/*.yml) los pasa como base_parameters;
# corriendo a mano se usan los valores por defecto.
dbutils.widgets.text("catalog", "catalogo_hackathon")
dbutils.widgets.text("schema", "hackathon")
dbutils.widgets.text("volume", "volume_hackathon")
CATALOG, SCHEMA, VOLUME = (
    dbutils.widgets.get(k) for k in ["catalog", "schema", "volume"]
)

BASE = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}")
DB = f"{CATALOG}.{SCHEMA}"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# Archivo esperado -> tabla Bronze. Se busca por prefijo para tolerar nombres con sufijos.
ARCHIVOS = {
    "01_market_prices_raw_expanded": "bronze_mercado_intradia",
    "01_market_prices_raw": "bronze_mercado_diario",
    "02_macro_raw": "bronze_macro",
    "03_fundamentals_raw": "bronze_fundamentales",
    "04_asset_reference": "bronze_activos",
    "05_client_profiles": "bronze_clientes",
    "06_events": "bronze_eventos",
}


def guardar(df, nombre):
    # Spark no infiere tipo de columnas 100% nulas: se guardan como texto vacío
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object and df[c].isna().all():
            df[c] = ""
    spark.createDataFrame(df).write.mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(f"{DB}.{nombre}")


def leer(nombre):
    return spark.table(f"{DB}.{nombre}").toPandas()


# COMMAND ----------

# ===== CELDA 1: LECTURA (Bronze): cada CSV tal cual, todo como texto =====
# Todo se lee como texto para no perder errores como "1,845.7" o "Q4-2024":
# si Bronze los convirtiera, Silver ya no podría explicar qué corrigió.
encontrados = {p.name: p for p in BASE.glob("*.csv")}
perfil = []
for prefijo, tabla in ARCHIVOS.items():
    # El prefijo más largo gana: "..._raw_expanded" no debe caer en "..._raw"
    candidatos = [
        n
        for n in encontrados
        if n.startswith(prefijo)
        and not any(
            n.startswith(o) for o in ARCHIVOS if o != prefijo and o.startswith(prefijo)
        )
    ]
    if not candidatos:
        print(f"FALTA en el volumen: {prefijo}*.csv -> {tabla} no se crea")
        continue
    ruta = encontrados[candidatos[0]]
    crudo = ruta.read_bytes()
    df = pd.read_csv(ruta, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df.insert(
        0, "_fila_origen", np.arange(2, len(df) + 2)
    )  # línea del CSV: trazabilidad fila a fila
    df["_archivo"] = ruta.name
    df["_hash_archivo"] = hashlib.sha256(crudo).hexdigest()
    df["_run_id"] = RUN_ID
    guardar(df, tabla)
    vacios = {
        c: int((df[c] == "").sum())
        for c in df.columns
        if not c.startswith("_") and (df[c] == "").any()
    }
    perfil.append(
        {
            "tabla": tabla,
            "archivo": ruta.name,
            "filas": len(df),
            "tiene_bom": crudo.startswith(b"\xef\xbb\xbf"),
            "vacios_por_columna": json.dumps(vacios),
        }
    )

df_perfil = pd.DataFrame(perfil)
guardar(df_perfil, "bronze_perfil_archivos")
display(df_perfil)

# COMMAND ----------

# ===== CELDA 2: HERRAMIENTAS DE LIMPIEZA (bitácora + marcas por fila) =====
BITACORA = []


def registrar(tabla, regla, accion, mask, porque, df, cols):
    # Una línea por regla: cuántas filas tocó, qué hizo, por qué y 3 ejemplos con su fila de origen
    mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
    ejemplo = (
        df.loc[mask, ["_fila_origen"] + cols].head(3).astype(str).to_dict("records")
    )
    BITACORA.append(
        {
            "tabla": tabla,
            "regla": regla,
            "accion": accion,
            "filas_afectadas": int(mask.sum()),
            "porque": porque,
            "ejemplos": json.dumps(ejemplo, ensure_ascii=False),
            "run_id": RUN_ID,
        }
    )
    print(f"[{tabla}] {regla}: {int(mask.sum())} filas -> {accion}")
    return mask


def marcar(df, mask, codigo):
    # dq_flags acumula los códigos de regla que afectaron a la fila
    mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
    df.loc[mask, "dq_flags"] = (df.loc[mask, "dq_flags"] + "," + codigo).str.strip(",")


def parsear_fechas(s, formatos_iso, formatos_alt):
    # Devuelve (fecha, era_formato_alterno). Prueba primero ISO; si falla, el formato alterno.
    iso = pd.Series(pd.NaT, index=s.index)
    for f in formatos_iso:
        iso = iso.fillna(pd.to_datetime(s, format=f, errors="coerce"))
    alt = pd.Series(pd.NaT, index=s.index)
    for f in formatos_alt:
        alt = alt.fillna(pd.to_datetime(s.where(iso.isna()), format=f, errors="coerce"))
    return iso.fillna(alt), iso.isna() & alt.notna()


def a_numero(s):
    # Quita separador de miles solo si tiene forma 1,234.5 (no adivina comas decimales)
    # "N/A", "NA", "null" son vacíos escritos como texto: se tratan igual que una celda vacía
    s = s.str.strip().replace(
        {"N/A": "", "NA": "", "n/a": "", "null": "", "NULL": "", "-": ""}
    )
    miles = s.str.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?")
    return pd.to_numeric(
        s.where(~miles, s.str.replace(",", "")).replace("", np.nan), errors="coerce"
    ), miles


activos = leer("bronze_activos")
ACTIVOS_VALIDOS = set(activos["asset_id"])
MONEDA_ACTIVO = dict(zip(activos["asset_id"], activos["currency"]))
MERCADO_ACTIVO = dict(zip(activos["asset_id"], activos["market"]))


def normalizar_activo(s):
    # VF-A1, VF_A6, VFA03 -> VF_A01, VF_A06, VF_A03
    m = s.str.extract(r"^VF[-_]?A0?(\d{1,2})$", expand=False)
    return ("VF_A" + m.str.zfill(2)).where(m.notna(), s)


# COMMAND ----------


# ===== CELDA 3: MERCADO (Silver): misma lógica para diario e intradía =====
def limpiar_mercado(tabla_bronze, tabla_silver, con_hora):
    if not spark.catalog.tableExists(f"{DB}.{tabla_bronze}"):
        print("No existe", tabla_bronze)
        return None
    raw = leer(tabla_bronze)
    df = raw.copy()
    df["dq_flags"] = ""
    df["dq_excluir"] = False
    t = tabla_silver

    # R1 · Fecha en formato MM/DD/YYYY. Se corrige: el dato es válido, solo cambia la escritura.
    # Se usa MM/DD (no DD/MM) porque los casos sin ambigüedad (día > 12, ej. 02/27/2026) lo prueban.
    iso = (
        ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]
        if con_hora
        else ["%Y-%m-%d"]
    )
    alt = (
        ["%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y"]
        if con_hora
        else ["%m/%d/%Y"]
    )
    df["fecha"], era_alt = parsear_fechas(df["date"], iso, alt)
    registrar(
        t,
        "R1_fecha_formato_mmdd",
        "CORREGIR",
        era_alt,
        "Mismo día escrito como MM/DD/YYYY; se convierte a ISO. Los casos con día > 12 confirman MM/DD.",
        df,
        ["date", "asset_id"],
    )
    marcar(df, era_alt, "R1")
    # Ambigüedad: si DD/MM también fuera una fecha válida dentro del rango del archivo, se deja marcada
    dia, mes = df["date"].str[3:5], df["date"].str[0:2]
    ambigua = era_alt & (pd.to_numeric(dia, errors="coerce") <= 12) & (dia != mes)
    # Se intercambian día y mes y se vuelve a leer como MM/DD
    alterna, _ = parsear_fechas(dia + "/" + mes + df["date"].str[5:], [], alt)
    dentro = ambigua & alterna.between(
        df.loc[~era_alt, "fecha"].min(), df.loc[~era_alt, "fecha"].max()
    )
    registrar(
        t,
        "R1b_fecha_ambigua",
        "MARCAR",
        dentro,
        "La lectura DD/MM también cae dentro del rango del archivo; se usa MM/DD por consistencia con R1 y se deja marcada.",
        df,
        ["date", "asset_id"],
    )
    marcar(df, dentro, "R1b")
    sin_fecha = df["fecha"].isna()
    registrar(
        t,
        "R1c_fecha_invalida",
        "EXCLUIR_ANALISIS",
        sin_fecha,
        "Fecha imposible de interpretar: no se puede ubicar el precio en el tiempo.",
        df,
        ["date", "asset_id"],
    )
    df.loc[sin_fecha, "dq_excluir"] = True
    marcar(df, sin_fecha, "R1c")

    # R2 · Código de activo mal escrito. Se corrige solo si el código normalizado existe en la
    # tabla de activos Y el mercado de la fila coincide con el del activo (doble verificación).
    df["asset_id_original"] = df["asset_id"]
    norm = normalizar_activo(df["asset_id"])
    cambio = (
        (norm != df["asset_id"])
        & norm.isin(ACTIVOS_VALIDOS)
        & (df["market"] == norm.map(MERCADO_ACTIVO))
    )
    registrar(
        t,
        "R2_codigo_activo",
        "CORREGIR",
        cambio,
        "Error de digitación del código (VF-A1, VFA03…). El código corregido existe en 04_asset_reference y su mercado coincide.",
        df,
        ["asset_id", "market"],
    )
    df.loc[cambio, "asset_id"] = norm[cambio]
    marcar(df, cambio, "R2")
    desconocido = ~df["asset_id"].isin(ACTIVOS_VALIDOS)
    registrar(
        t,
        "R2b_activo_desconocido",
        "EXCLUIR_ANALISIS",
        desconocido,
        "Activo que no existe en la tabla de referencia: no se puede asignar sector, país ni emisor.",
        df,
        ["asset_id"],
    )
    df.loc[desconocido, "dq_excluir"] = True
    marcar(df, desconocido, "R2b")

    # Conversión numérica (lo que no convierte queda nulo y lo capturan las reglas siguientes)
    # R2c · Volumen abreviado ("3635.6K", "1.2M"). Se corrige: es el mismo número en notación
    # abreviada (K = miles, M = millones); sin esto quedaría vacío sin motivo.
    sufijo = df["volume"].str.strip().str.extract(r"^([\d.]+)\s*([KkMm])$")
    abreviado = sufijo[0].notna()
    registrar(
        t,
        "R2c_volumen_abreviado",
        "CORREGIR",
        abreviado,
        "Volumen escrito con sufijo K/M; se expande al número completo (K = ×1 000, M = ×1 000 000).",
        df,
        ["date", "asset_id", "volume"],
    )
    factor = sufijo[1].str.upper().map({"K": 1e3, "M": 1e6})
    df.loc[abreviado, "volume"] = (
        (pd.to_numeric(sufijo.loc[abreviado, 0]) * factor[abreviado])
        .round()
        .astype("int64")
        .astype(str)
    )
    marcar(df, abreviado, "R2c")

    for c in ["open", "high", "low", "close", "volume"]:
        df[c], _ = a_numero(df[c])

    # R3 · Duplicados exactos. Contar dos veces la misma vela duplicaría su peso en los retornos.
    llave = ["fecha", "asset_id"]
    negocio = llave + ["market", "currency", "open", "high", "low", "close", "volume"]
    dup_exacto = df.duplicated(negocio, keep="first")
    registrar(
        t,
        "R3_duplicado_exacto",
        "ELIMINAR_DUPLICADO",
        dup_exacto,
        "Fila idéntica a otra (misma fecha, activo y valores). Se conserva una.",
        df,
        ["date", "asset_id", "close"],
    )
    df = df[~dup_exacto].copy()
    # Misma llave pero valores distintos: no hay forma de saber cuál es la buena -> ambas fuera del análisis
    conflicto = df.duplicated(llave, keep=False)
    registrar(
        t,
        "R3b_duplicado_conflictivo",
        "EXCLUIR_ANALISIS",
        conflicto,
        "Misma fecha y activo con precios distintos; ninguna versión es verificable.",
        df,
        ["date", "asset_id", "close"],
    )
    df.loc[conflicto, "dq_excluir"] = True
    marcar(df, conflicto, "R3b")

    # R4 · Moneda vacía. Se completa desde la tabla de activos: todos cotizan en USD según la referencia.
    sin_moneda = df["currency"].str.strip() == ""
    registrar(
        t,
        "R4_moneda_vacia",
        "CORREGIR",
        sin_moneda,
        "La moneda es atributo del activo (04_asset_reference); se completa desde ahí.",
        df,
        ["asset_id", "currency"],
    )
    df.loc[sin_moneda, "currency"] = df.loc[sin_moneda, "asset_id"].map(MONEDA_ACTIVO)
    marcar(df, sin_moneda, "R4")

    # R5 · Volumen vacío. Se MARCA y no se imputa: el volumen no entra al retorno y
    # un valor inventado distorsionaría cualquier lectura de liquidez.
    sin_vol = df["volume"].isna()
    registrar(
        t,
        "R5_volumen_vacio",
        "MARCAR",
        sin_vol,
        "Volumen faltante. No afecta retornos; no se imputa para no inventar liquidez.",
        df,
        ["date", "asset_id"],
    )
    marcar(df, sin_vol, "R5")

    # R6 · Cierre vacío. La fila se conserva (open/high/low son válidos) pero no sirve para retornos.
    sin_close = df["close"].isna()
    registrar(
        t,
        "R6_cierre_vacio",
        "EXCLUIR_ANALISIS",
        sin_close,
        "Sin precio de cierre no hay retorno. Imputarlo sería inventar el dato que se quiere medir.",
        df,
        ["date", "asset_id", "open", "high", "low"],
    )
    df.loc[sin_close, "dq_excluir"] = True
    marcar(df, sin_close, "R6")

    # R7 · Error de escala: cierre fuera de [low, high]. Se corrige solo si dividir/multiplicar por
    # 10 o 100 lo devuelve dentro del rango del mismo día (ej. 3589 -> 35.89 con rango 34.56–35.92).
    fuera = df["close"].notna() & (
        (df["close"] > df["high"] * 1.001) | (df["close"] < df["low"] * 0.999)
    )
    corregido = pd.Series(np.nan, index=df.index)
    for f in [0.01, 0.1, 10, 100]:
        c = df["close"] * f
        ok = fuera & corregido.isna() & c.between(df["low"] * 0.999, df["high"] * 1.001)
        corregido[ok] = c[ok]
    escala = corregido.notna()
    registrar(
        t,
        "R7_cierre_error_escala",
        "CORREGIR",
        escala,
        "Cierre fuera del rango del día; con el punto decimal corrido cae dentro del rango. Se corrige la escala.",
        df,
        ["date", "asset_id", "low", "high", "close"],
    )
    df.loc[escala, "close"] = corregido[escala].round(4)
    marcar(df, escala, "R7")
    incoherente = fuera & ~escala
    registrar(
        t,
        "R7b_cierre_incoherente",
        "EXCLUIR_ANALISIS",
        incoherente,
        "Cierre fuera del rango del día sin corrección verificable.",
        df,
        ["date", "asset_id", "low", "high", "close"],
    )
    df.loc[incoherente, "dq_excluir"] = True
    marcar(df, incoherente, "R7b")

    # R8 · Coherencia OHLC (low ≤ open ≤ high, low ≤ high)
    ohlc_mal = (
        (df["low"] > df["high"])
        | (df["open"] > df["high"] * 1.001)
        | (df["open"] < df["low"] * 0.999)
    )
    registrar(
        t,
        "R8_ohlc_incoherente",
        "EXCLUIR_ANALISIS",
        ohlc_mal,
        "Precios del día que se contradicen entre sí.",
        df,
        ["date", "asset_id", "open", "high", "low", "close"],
    )
    df.loc[ohlc_mal, "dq_excluir"] = True
    marcar(df, ohlc_mal, "R8")

    # R9 · Saltos extremos entre observaciones consecutivas. Solo se MARCAN: pueden ser eventos reales
    # (06_events) y quitarlos borraría justamente la señal que se busca.
    df = df.sort_values(["asset_id", "fecha"])
    validos = ~df["dq_excluir"]
    ret = df[validos].groupby("asset_id")["close"].pct_change()
    salto = pd.Series(False, index=df.index)
    salto[ret.index] = ret.abs() > (0.10 if con_hora else 0.20)
    registrar(
        t,
        "R9_salto_extremo",
        "MARCAR",
        salto,
        "Variación consecutiva atípica. Se conserva: puede ser un evento real; el modelador la revisa contra 06_events.",
        df,
        ["date", "asset_id", "close"],
    )
    marcar(df, salto, "R9")

    df["dq_estado"] = np.where(
        df["dq_excluir"],
        "EXCLUIR_ANALISIS",
        np.where(df["dq_flags"] == "", "OK", "CORREGIDO_O_MARCADO"),
    )
    cols = [
        "fecha",
        "asset_id",
        "asset_id_original",
        "market",
        "currency",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dq_estado",
        "dq_flags",
        "dq_excluir",
        "_fila_origen",
        "_archivo",
    ]
    out = df[cols].reset_index(drop=True)
    guardar(out, tabla_silver)
    print(tabla_silver, out["dq_estado"].value_counts().to_dict())
    return out


silver_diario = limpiar_mercado(
    "bronze_mercado_diario", "silver_mercado_diario", con_hora=False
)
silver_intradia = limpiar_mercado(
    "bronze_mercado_intradia", "silver_mercado_intradia", con_hora=True
)

# COMMAND ----------

# ===== CELDA 4: INTRADÍA -> VELA DIARIA (Silver conformado) =====
# Se agrega la vela de 10 minutos a vela diaria para poder compararla con el archivo diario
# y para que Gold trabaje con la misma granularidad que macro (diaria).
if silver_intradia is not None:
    v = silver_intradia[~silver_intradia["dq_excluir"]].sort_values("fecha").copy()
    v["dia"] = v["fecha"].dt.normalize()
    agg = (
        v.groupby(["dia", "asset_id"])
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            n_velas=("close", "size"),
        )
        .reset_index()
        .rename(columns={"dia": "fecha"})
    )
    # Día con pocas velas: el cierre puede no ser el de fin de jornada. Se marca, no se elimina.
    tipico = agg["n_velas"].median()
    agg["dia_incompleto"] = agg["n_velas"] < 0.5 * tipico
    BITACORA.append(
        {
            "tabla": "silver_mercado_intradia_diario",
            "regla": "R10_dia_incompleto",
            "accion": "MARCAR",
            "filas_afectadas": int(agg["dia_incompleto"].sum()),
            "porque": f"Día con menos de la mitad de las velas típicas ({tipico:.0f}); su cierre puede no ser el de fin de jornada.",
            "ejemplos": agg.loc[agg["dia_incompleto"], ["fecha", "asset_id", "n_velas"]]
            .head(3)
            .astype(str)
            .to_json(orient="records"),
            "run_id": RUN_ID,
        }
    )
    guardar(agg, "silver_mercado_intradia_diario")
    display(agg.groupby("asset_id")["fecha"].agg(["min", "max", "count"]))

# COMMAND ----------

# ===== CELDA 5: MACRO (Silver) =====
raw = leer("bronze_macro")
df = raw.copy()
df["dq_flags"] = ""
t = "silver_macro"
df["fecha"], _ = parsear_fechas(df["date"], ["%Y-%m-%d"], ["%m/%d/%Y"])
for c in [
    "us_10y_yield",
    "us_inflation_yoy",
    "usd_cop",
    "usd_pen",
    "usd_mxn",
    "usd_clp",
    "market_factor",
]:
    df[c], _ = a_numero(df[c])
df = df.sort_values("fecha")

# M1 · La inflación es un dato MENSUAL publicado el primer día hábil del mes; el resto de días
# viene vacío porque no hubo publicación, no porque falte. Se arrastra el último valor publicado
# a una columna NUEVA y se conserva la original para saber qué días hubo publicación.
pub = df["us_inflation_yoy"].notna()
registrar(
    t,
    "M1_inflacion_mensual",
    "CORREGIR",
    ~pub,
    "Dato mensual: se completa con el último valor publicado (sin mirar al futuro). La columna original se conserva.",
    df,
    ["date", "us_inflation_yoy"],
)
df["us_inflation_yoy_vigente"] = df["us_inflation_yoy"].ffill()
df["inflacion_publicada_hoy"] = pub
marcar(df, ~pub, "M1")

# M2 · Régimen de mercado con valores fuera del dominio esperado
fuera = ~df["risk_regime"].isin(["Normal", "Stress", "Recovery"])
registrar(
    t,
    "M2_regimen_desconocido",
    "MARCAR",
    fuera,
    "Régimen fuera de {Normal, Stress, Recovery}.",
    df,
    ["date", "risk_regime"],
)
marcar(df, fuera, "M2")

# M3 · Variables numéricas vacías (FX, tasa, factor de mercado)
for c in ["us_10y_yield", "usd_cop", "usd_pen", "market_factor"]:
    m = df[c].isna()
    if m.any():
        registrar(
            t,
            f"M3_{c}_vacio",
            "MARCAR",
            m,
            "Dato diario faltante; no se imputa en Silver.",
            df,
            ["date", c],
        )
        marcar(df, m, "M3")

silver_macro = df.drop(columns=["date"]).rename(columns={"fecha": "fecha"})
silver_macro = silver_macro[
    [
        "fecha",
        "us_10y_yield",
        "us_inflation_yoy",
        "us_inflation_yoy_vigente",
        "inflacion_publicada_hoy",
        "usd_cop",
        "usd_pen",
        "usd_mxn",
        "usd_clp",
        "market_factor",
        "risk_regime",
        "dq_flags",
        "_fila_origen",
    ]
]
guardar(silver_macro, t)

# COMMAND ----------

# ===== CELDA 6: FUNDAMENTALES (Silver) =====
raw = leer("bronze_fundamentales")
df = raw.copy()
df["dq_flags"] = ""
df["dq_excluir"] = False
t = "silver_fundamentales"
EMISORES = set(activos["issuer_id"])

# F1 · Código de emisor mal escrito (ISS04 -> ISS_04), verificado contra la tabla de activos
df["issuer_id_original"] = df["issuer_id"]
num = df["issuer_id"].str.extract(r"^ISS_?(\d{1,2})$", expand=False)
norm = ("ISS_" + num.str.zfill(2)).where(num.notna(), df["issuer_id"])
cambio = (norm != df["issuer_id"]) & norm.isin(EMISORES)
registrar(
    t,
    "F1_codigo_emisor",
    "CORREGIR",
    cambio,
    "Error de digitación; el código corregido existe en 04_asset_reference.",
    df,
    ["issuer_id", "period_end"],
)
df.loc[cambio, "issuer_id"] = norm[cambio]
marcar(df, cambio, "F1")

# F2 · Número con separador de miles guardado como texto ("1,845.7")
for c in [
    "revenue_usd_m",
    "ebitda_usd_m",
    "net_income_usd_m",
    "debt_usd_m",
    "cash_usd_m",
]:
    df[c], miles = a_numero(df[c])
    if miles.any():
        registrar(
            t,
            f"F2_{c}_separador_miles",
            "CORREGIR",
            miles,
            "Separador de miles en un campo numérico; el valor no cambia.",
            df,
            ["issuer_id", "period_end", c],
        )
        marcar(df, miles, "F2")

# F3 · Periodo en formato trimestral ("Q4-2024")
df["periodo"], _ = parsear_fechas(df["period_end"], ["%Y-%m-%d"], [])
trim = df["period_end"].str.fullmatch(r"Q[1-4]-\d{4}")
q = df.loc[trim, "period_end"].str.extract(r"Q(\d)-(\d{4})").astype(int)
df.loc[trim, "periodo"] = [
    pd.Timestamp(y, 3 * n, 1) + pd.offsets.MonthEnd(0) for n, y in zip(q[0], q[1])
]
registrar(
    t,
    "F3_periodo_trimestral",
    "CORREGIR",
    trim,
    "Periodo escrito como Qn-AAAA; se convierte al cierre de trimestre.",
    df,
    ["issuer_id", "period_end"],
)
marcar(df, trim, "F3")

# F3b · ¿La conversión choca con un trimestre que ya existe con OTROS valores?
# Caso ISS_02: "Q4-2024" chocaría con 2024-12-31 (valores distintos), al emisor le falta
# justamente 2025-06-30 y la fila está ubicada en el archivo entre 2025-03-31 y 2025-09-30.
# Si las tres evidencias coinciden, se reasigna al trimestre faltante y se marca como INFERIDO.
cols_val = [
    "revenue_usd_m",
    "ebitda_usd_m",
    "net_income_usd_m",
    "debt_usd_m",
    "cash_usd_m",
]
for i in df.index[trim]:
    e, p = df.at[i, "issuer_id"], df.at[i, "periodo"]
    otras = df[(df["issuer_id"] == e) & (df.index != i)]
    choca = otras[(otras["periodo"] == p)]
    if choca.empty or (choca[cols_val].values == df.loc[[i], cols_val].values).all():
        continue
    esperados = pd.date_range(
        df.loc[df["issuer_id"] == e, "periodo"].min(),
        df["periodo"].max(),
        freq=pd.offsets.QuarterEnd(),
    )
    faltan = sorted(set(esperados) - set(otras["periodo"]))
    previo = otras[otras["_fila_origen"].astype(int) < int(df.at[i, "_fila_origen"])][
        "periodo"
    ].max()
    siguiente = otras[
        otras["_fila_origen"].astype(int) > int(df.at[i, "_fila_origen"])
    ]["periodo"].min()
    hueco = [f for f in faltan if previo < f < siguiente]
    m = pd.Series(df.index == i, index=df.index)
    if len(hueco) == 1:
        registrar(
            t,
            "F3b_periodo_inferido",
            "CORREGIR",
            m,
            f"'{df.at[i, 'period_end']}' chocaría con {p.date()} (valores distintos). Al emisor le falta {hueco[0].date()} "
            f"y la fila está entre {previo.date()} y {siguiente.date()} en el archivo: se asigna {hueco[0].date()} y se marca INFERIDO.",
            df,
            ["issuer_id", "period_end", "revenue_usd_m"],
        )
        df.at[i, "periodo"] = hueco[0]
        marcar(df, m, "F3b_INFERIDO")
    else:
        registrar(
            t,
            "F3c_periodo_no_resoluble",
            "EXCLUIR_ANALISIS",
            m,
            "Choca con un trimestre existente y no hay hueco único que lo explique.",
            df,
            ["issuer_id", "period_end"],
        )
        df.at[i, "dq_excluir"] = True
        marcar(df, m, "F3c")

# F4 · Duplicado exacto (mismo emisor, periodo y valores)
dup = df.duplicated(["issuer_id", "periodo"] + cols_val, keep="first")
registrar(
    t,
    "F4_duplicado_exacto",
    "ELIMINAR_DUPLICADO",
    dup,
    "Fila repetida; contarla dos veces sesga los totales de 12 meses.",
    df,
    ["issuer_id", "period_end"],
)
df = df[~dup].copy()

# F5 · Caja vacía. Se MARCA: interpolar caja inventaría liquidez de la empresa.
sin_caja = df["cash_usd_m"].isna()
registrar(
    t,
    "F5_caja_vacia",
    "MARCAR",
    sin_caja,
    "Dato faltante; no se imputa. Gold usa el último trimestre con dato si lo necesita.",
    df,
    ["issuer_id", "period_end"],
)
marcar(df, sin_caja, "F5")

# F6 · Serie incompleta por emisor (para que Gold no compare crecimientos con distinta base)
n_trim = df[~df["dq_excluir"]].groupby("issuer_id")["periodo"].nunique()
df["trimestres_emisor"] = df["issuer_id"].map(n_trim)
df["serie_completa"] = df["trimestres_emisor"] == n_trim.max()

silver_fund = df[
    ["issuer_id", "issuer_id_original", "period_end", "periodo"]
    + cols_val
    + ["trimestres_emisor", "serie_completa", "dq_flags", "dq_excluir", "_fila_origen"]
].sort_values(["issuer_id", "periodo"])
guardar(silver_fund, t)
display(silver_fund[silver_fund["dq_flags"] != ""])

# COMMAND ----------

# ===== CELDA 7: REFERENCIAS: ACTIVOS, CLIENTES, EVENTOS (Silver) =====
guardar(activos.drop(columns=["_hash_archivo"]), "silver_activos")

# Clientes: se tipan los números; el texto libre (constraint, priority) se conserva tal cual
# porque lo traduce a reglas el agente en el notebook 02.
cli = leer("bronze_clientes")
cli["horizon_months"] = pd.to_numeric(cli["horizon_months"])
cli["max_drawdown"] = pd.to_numeric(cli["max_drawdown_tolerance_pct"]) / 100
guardar(cli.drop(columns=["_hash_archivo"]), "silver_clientes")

# Eventos: "scope" mezcla un activo (VF_A02) con un sector (Technology). Se separa en tipo y valor.
ev = leer("bronze_eventos")
t = "silver_eventos"
ev["dq_flags"] = ""
ev["fecha"], _ = parsear_fechas(ev["event_date"], ["%Y-%m-%d"], ["%m/%d/%Y"])
ev["scope_tipo"] = np.select(
    [ev["scope"].isin(ACTIVOS_VALIDOS), ev["scope"].isin(set(activos["sector"]))],
    ["activo", "sector"],
    "desconocido",
)
registrar(
    t,
    "E1_scope_mixto",
    "CORREGIR",
    ev["scope_tipo"] == "sector",
    "El alcance es un sector, no un activo; se separa para poder unirlo a los activos de ese sector.",
    ev,
    ["event_date", "scope"],
)
desc = ev["scope_tipo"] == "desconocido"
registrar(
    t,
    "E2_scope_desconocido",
    "MARCAR",
    desc,
    "Alcance que no es activo ni sector conocido.",
    ev,
    ["scope"],
)
marcar(ev, desc, "E2")
# Una fila por activo afectado: facilita medir el impacto del evento en Gold
filas = []
for r in ev.itertuples():
    afectados = (
        [r.scope]
        if r.scope_tipo == "activo"
        else activos.loc[activos["sector"] == r.scope, "asset_id"].tolist()
    )
    for a in afectados:
        filas.append(
            {
                "fecha": r.fecha,
                "asset_id": a,
                "scope": r.scope,
                "scope_tipo": r.scope_tipo,
                "event_type": r.event_type,
                "headline": r.headline,
                "severity": r.severity,
                "dq_flags": r.dq_flags,
            }
        )
guardar(pd.DataFrame(filas), t)
display(pd.DataFrame(filas))

# COMMAND ----------

# ===== CELDA 8: ¿QUÉ FUENTE DE PRECIOS ES CONFIABLE? (Find the truth) =====
# Hay dos archivos de mercado. Antes de modelar hay que saber si cuentan la misma historia.
if silver_diario is not None and silver_intradia is not None:
    d = silver_diario[~silver_diario["dq_excluir"]][["fecha", "asset_id", "close"]]
    i = leer("silver_mercado_intradia_diario")[["fecha", "asset_id", "close"]]
    c = d.merge(i, on=["fecha", "asset_id"], suffixes=("_diario", "_intradia"))
    c["dif_pct"] = (c["close_intradia"] / c["close_diario"] - 1) * 100
    c = c.sort_values(["asset_id", "fecha"])
    for s in ["diario", "intradia"]:
        c[f"ret_{s}"] = c.groupby("asset_id")[f"close_{s}"].pct_change()
    resumen = (
        c.groupby("asset_id")
        .apply(
            lambda g: pd.Series(
                {
                    "dias_comunes": len(g),
                    "dif_mediana_pct": g["dif_pct"].abs().median(),
                    "ret_periodo_diario_pct": (
                        g["close_diario"].iloc[-1] / g["close_diario"].iloc[0] - 1
                    )
                    * 100,
                    "ret_periodo_intradia_pct": (
                        g["close_intradia"].iloc[-1] / g["close_intradia"].iloc[0] - 1
                    )
                    * 100,
                    "corr_retornos": g["ret_diario"].corr(g["ret_intradia"]),
                }
            )
        )
        .reset_index()
    )
    resumen["misma_historia"] = (resumen["dif_mediana_pct"] < 1) & (
        resumen["corr_retornos"] > 0.9
    )
    guardar(resumen, "silver_conciliacion_fuentes")
    BITACORA.append(
        {
            "tabla": "silver_conciliacion_fuentes",
            "regla": "C1_fuentes_precio",
            "accion": "MARCAR",
            "filas_afectadas": int((~resumen["misma_historia"]).sum()),
            "porque": "Activos donde el archivo diario y el intradía no cuentan la misma historia. "
            "Se conservan ambas fuentes; Gold usa FUENTE_PRECIOS y reporta la otra como sensibilidad.",
            "ejemplos": resumen.head(3).round(2).to_json(orient="records"),
            "run_id": RUN_ID,
        }
    )
    display(resumen.round(2))

# COMMAND ----------

# ===== CELDA 9: BITÁCORA DE CALIDAD (evidencia para el jurado) =====
df_bit = pd.DataFrame(BITACORA)
guardar(df_bit, "silver_bitacora_calidad")
display(df_bit[["tabla", "regla", "accion", "filas_afectadas", "porque"]])
print(
    "Resumen por acción:", df_bit.groupby("accion")["filas_afectadas"].sum().to_dict()
)
