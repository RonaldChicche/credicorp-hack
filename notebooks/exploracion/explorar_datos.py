# Databricks notebook source
# MAGIC %md
# MAGIC # Exploración y limpieza de datos
# MAGIC Lee los CSV crudos del volumen `/Volumes/<catalog>/<schema>/raw`, documenta los problemas de
# MAGIC calidad, los corrige y escribe tablas Delta `*_clean` + `dq_log` (qué se corrigió y cuántas filas).

# COMMAND ----------

import re
import numpy as np
import pandas as pd

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "credicorp_hack")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
RAW = f"/Volumes/{catalog}/{schema}/raw"
TARGET = f"{catalog}.{schema}"

dq_log = []  # (tabla, regla, filas_afectadas)


def log(tabla, regla, n):
    dq_log.append((tabla, regla, int(n)))
    print(f"[{tabla}] {regla}: {int(n)}")


def read_csv(name):
    # utf-8-sig: los CSV traen BOM
    return pd.read_csv(f"{RAW}/{name}", encoding="utf-8-sig")


# COMMAND ----------

# MAGIC %md ## 1. Carga y diagnóstico del dato crudo

# COMMAND ----------

prices = read_csv("01_market_prices_raw.csv")
intraday = read_csv("01_market_prices_raw_expanded.csv")
macro = read_csv("02_macro_raw.csv")
fund = read_csv("03_fundamentals_raw.csv")
assets = read_csv("04_asset_reference.csv")
clients = read_csv("05_client_profiles.csv")
events = read_csv("06_events.csv")

raw = {"prices": prices, "intraday": intraday, "macro": macro, "fundamentals": fund,
       "assets": assets, "clients": clients, "events": events}

diag = pd.DataFrame([
    {"tabla": k, "filas": len(v), "columnas": v.shape[1],
     "nulos": int(v.isna().sum().sum()), "duplicados_exactos": int(v.duplicated().sum())}
    for k, v in raw.items()
])
display(diag)

# COMMAND ----------

# Problemas detectados a simple vista
print("asset_id en precios :", sorted(prices.asset_id.unique()))
print("asset_id en intradía:", sorted(intraday.asset_id.unique()))
print("issuer_id fundamentales:", sorted(fund.issuer_id.unique()))
print("fechas no ISO en precios:", prices[~prices.date.astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")].date.unique())
print("volumen no numérico:", prices[pd.to_numeric(prices.volume, errors="coerce").isna()].volume.unique())
print("revenue no numérico:", fund[pd.to_numeric(fund.revenue_usd_m, errors="coerce").isna()].revenue_usd_m.unique())
print("nulos intradía por columna:", intraday.isna().sum()[lambda s: s > 0].to_dict())

# COMMAND ----------

# MAGIC %md ## 2. Funciones de limpieza

# COMMAND ----------

def norm_asset(x):
    """VF-A1 / VFA03 / VF_A6 -> VF_A01 / VF_A03 / VF_A06"""
    m = re.match(r"^VF[\s_-]*A0*(\d+)$", str(x).strip().upper())
    return f"VF_A{int(m.group(1)):02d}" if m else None


def norm_issuer(x):
    """ISS04 -> ISS_04"""
    m = re.match(r"^ISS[\s_-]*0*(\d+)$", str(x).strip().upper())
    return f"ISS_{int(m.group(1)):02d}" if m else None


def parse_dates(s, iso_fmt, us_fmt):
    """ISO primero; lo que no calce se interpreta como MM/DD/YYYY (p. ej. 02/27/2026 solo es válido así)."""
    s = s.astype("object").map(lambda x: x.strip() if isinstance(x, str) else x)
    out = pd.to_datetime(s, format=iso_fmt, errors="coerce")
    miss = out.isna() & s.notna()
    if miss.any():
        out[miss] = pd.to_datetime(s[miss], format=us_fmt, errors="coerce")
    return out, int(miss.sum())


def parse_volume(v):
    """'3635.6K' -> 3635600; NaN se conserva."""
    if pd.isna(v):
        return np.nan
    t = str(v).strip().replace(",", "").upper()
    mult = 1
    if t.endswith("K"):
        mult, t = 1e3, t[:-1]
    elif t.endswith("M"):
        mult, t = 1e6, t[:-1]
    return round(float(t) * mult)


def fix_close_scale(df, tabla):
    """Si el cierre queda fuera de [low, high], prueba /10, /100, /1000 (error de escala/decimal)."""
    bad = (df.close > df.high * 1.001) | (df.close < df.low * 0.999)
    fixed = 0
    unfixable = 0
    for i in df.index[bad]:
        r = df.loc[i]
        for f in (10, 100, 1000):
            c = r.close / f
            if r.low * 0.999 <= c <= r.high * 1.001:
                df.at[i, "close"] = round(c, 2)
                fixed += 1
                break
        else:
            df.at[i, "close"] = np.nan
            unfixable += 1
    log(tabla, "cierre fuera de [low, high] corregido por escala", fixed)
    log(tabla, "cierre fuera de [low, high] sin corrección (anulado)", unfixable)
    return df


def clean_prices(df, tabla, iso_fmt, us_fmt, ref):
    df = df.copy()
    n0 = len(df)

    df["asset_id"], orig = df.asset_id.map(norm_asset), df.asset_id
    log(tabla, "asset_id normalizado", (df.asset_id != orig).sum())
    log(tabla, "asset_id no reconocido (descartado)", df.asset_id.isna().sum())
    df = df.dropna(subset=["asset_id"])

    df["date"], n_us = parse_dates(df["date"], iso_fmt, us_fmt)
    log(tabla, "fecha en formato MM/DD/YYYY convertida", n_us)
    log(tabla, "fecha inválida (descartada)", df["date"].isna().sum())
    df = df.dropna(subset=["date"])

    if df.volume.dtype == object or str(df.volume.dtype).startswith(("str", "string")):
        non_num = pd.to_numeric(df.volume, errors="coerce").isna() & df.volume.notna()
        log(tabla, "volumen con sufijo K/M convertido", non_num.sum())
    df["volume"] = df.volume.map(parse_volume).astype("Int64")
    log(tabla, "volumen nulo (se conserva nulo)", df.volume.isna().sum())

    # market y currency se toman de la referencia de activos (fuente única)
    ref_map = ref.set_index("asset_id")
    log(tabla, "currency nulo rellenado desde referencia", df.currency.isna().sum())
    df["currency"] = df.asset_id.map(ref_map.currency)
    df["market"] = df.asset_id.map(ref_map.market)

    df = fix_close_scale(df, tabla)
    n_ohlc = df[["open", "high", "low", "close"]].isna().any(axis=1).sum()
    log(tabla, "filas sin OHLC completo (descartadas)", n_ohlc)
    df = df.dropna(subset=["open", "high", "low", "close"])

    before = len(df)
    df = df.sort_values(["asset_id", "date"]).drop_duplicates(["date", "asset_id"], keep="first")
    log(tabla, "duplicados (date, asset_id) eliminados", before - len(df))
    log(tabla, "filas finales / originales", len(df))
    print(f"   {tabla}: {n0} -> {len(df)} filas")
    return df.reset_index(drop=True)


# COMMAND ----------

# MAGIC %md ## 3. Limpieza por tabla

# COMMAND ----------

# Referencias
assets_c = assets.copy()
assets_c["asset_id"] = assets_c.asset_id.map(norm_asset)
assets_c["issuer_id"] = assets_c.issuer_id.map(norm_issuer)

# Precios diarios e intradía
prices_c = clean_prices(prices, "prices_daily", "%Y-%m-%d", "%m/%d/%Y", assets_c)
intraday_c = clean_prices(intraday, "prices_intraday", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S", assets_c)

# COMMAND ----------

# Macro: fechas, regimen, inflación (publicada mensual -> ffill con bandera)
macro_c = macro.copy()
macro_c["date"], _ = parse_dates(macro_c["date"], "%Y-%m-%d", "%m/%d/%Y")
macro_c = macro_c.sort_values("date").drop_duplicates("date")
macro_c["inflation_is_filled"] = macro_c.us_inflation_yoy.isna()
log("macro", "inflación mensual propagada (ffill) a días sin dato", macro_c.inflation_is_filled.sum())
macro_c["us_inflation_yoy"] = macro_c.us_inflation_yoy.ffill()
macro_c["risk_regime"] = macro_c.risk_regime.str.strip().str.title()

# Fundamentales
fund_c = fund.copy()
fund_c["issuer_id"], orig = fund_c.issuer_id.map(norm_issuer), fund_c.issuer_id
log("fundamentals", "issuer_id normalizado", (fund_c.issuer_id != orig).sum())
fund_c["revenue_usd_m"] = pd.to_numeric(fund_c.revenue_usd_m.astype(str).str.replace(",", "", regex=False), errors="coerce")
log("fundamentals", "revenue con separador de miles convertido", fund.revenue_usd_m.astype(str).str.contains(",").sum())


def parse_period(x):
    """'Q4-2024' -> 2024-12-31 (fin de trimestre); fechas ISO se respetan."""
    m = re.match(r"^Q([1-4])[-\s]?(\d{4})$", str(x).strip().upper())
    if m:
        return pd.Timestamp(int(m.group(2)), int(m.group(1)) * 3, 1) + pd.offsets.MonthEnd(0)
    return pd.to_datetime(x)


log("fundamentals", "period_end en formato Qn-YYYY convertido", fund_c.period_end.astype(str).str.upper().str.startswith("Q").sum())
fund_c["period_end"] = fund_c.period_end.map(parse_period)
before = len(fund_c)
fund_c = fund_c.drop_duplicates(["issuer_id", "period_end"], keep="first")
log("fundamentals", "duplicados (issuer, periodo) eliminados", before - len(fund_c))
fund_c = fund_c.sort_values(["issuer_id", "period_end"]).reset_index(drop=True)
fund_c["cash_is_imputed"] = fund_c.cash_usd_m.isna()
fund_c["cash_usd_m"] = fund_c.groupby("issuer_id").cash_usd_m.transform(lambda s: s.interpolate(limit_direction="both"))
log("fundamentals", "cash nulo imputado por interpolación dentro del emisor", fund_c.cash_is_imputed.sum())

# Eventos
events_c = events.copy()
events_c["event_date"] = pd.to_datetime(events_c.event_date)

# COMMAND ----------

# MAGIC %md ## 4. Validaciones post-limpieza

# COMMAND ----------

checks = {
    "precios: sin duplicados": not prices_c.duplicated(["date", "asset_id"]).any(),
    "precios: high >= low": bool((prices_c.high >= prices_c.low).all()),
    "precios: close dentro de [low, high]": bool(((prices_c.close <= prices_c.high * 1.001) & (prices_c.close >= prices_c.low * 0.999)).all()),
    "precios: 8 activos canónicos": prices_c.asset_id.nunique() == 8,
    "precios: todos los activos están en la referencia": set(prices_c.asset_id) <= set(assets_c.asset_id),
    "intradía: sin duplicados": not intraday_c.duplicated(["date", "asset_id"]).any(),
    "intradía: close dentro de [low, high]": bool(((intraday_c.close <= intraday_c.high * 1.001) & (intraday_c.close >= intraday_c.low * 0.999)).all()),
    "fundamentales: 8 emisores": fund_c.issuer_id.nunique() == 8,
    "fundamentales: sin nulos": not fund_c.drop(columns=[]).isna().any().any(),
    "macro: fechas únicas": macro_c.date.is_unique,
}
display(pd.DataFrame({"validación": list(checks), "ok": list(checks.values())}))
assert all(checks.values()), "Falló una validación de calidad"

# COMMAND ----------

# MAGIC %md ## 5. Escritura de tablas Delta

# COMMAND ----------

def save(df, name):
    sdf = spark.createDataFrame(df)
    sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{TARGET}.{name}")
    print(f"{TARGET}.{name}: {sdf.count()} filas")


spark.sql(f"CREATE SCHEMA IF NOT EXISTS {TARGET}")
save(prices_c, "prices_daily_clean")
save(intraday_c, "prices_intraday_clean")
save(macro_c, "macro_clean")
save(fund_c, "fundamentals_clean")
save(assets_c, "assets")
save(clients, "clients")
save(events_c, "events")
save(pd.DataFrame(dq_log, columns=["tabla", "regla", "filas_afectadas"]), "dq_log")

# COMMAND ----------

# MAGIC %md ## 6. Resumen de la limpieza

# COMMAND ----------

display(spark.table(f"{TARGET}.dq_log").filter("filas_afectadas > 0"))

# COMMAND ----------

display(spark.sql(f"SHOW TABLES IN {TARGET}"))
