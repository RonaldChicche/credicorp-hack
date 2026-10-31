# Databricks notebook source
# MAGIC %md
# MAGIC # Score de carteras por cliente
# MAGIC 1. **Filtro:** una cartera que incumple un límite duro queda no apta y no recibe score total.
# MAGIC 2. **Score:** cada indicador se lleva a 0–100 entre `ancla_0` y `ancla_100`; el grupo es el promedio de sus indicadores; el total pondera los grupos.
# MAGIC
# MAGIC Todo lo configurable (límites, pesos, anclas) viene de `config_*`. Salidas: `gold_score_results`, `gold_score_detail`,
# MAGIC `gold_limit_checks`, `gold_portfolio_weights`.

# COMMAND ----------

import numpy as np
import pandas as pd

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "credicorp_hack")
T = f"{dbutils.widgets.get('catalog')}.{dbutils.widgets.get('schema')}"
TD = 252


def vigente(df, keys):
    return df.sort_values("vigente_desde").drop_duplicates(keys, keep="last")


lim = vigente(spark.table(f"{T}.config_limits").toPandas(), ["cliente", "regla"])
pesos = vigente(spark.table(f"{T}.config_weights").toPandas(), ["cliente", "grupo"])
ind = vigente(spark.table(f"{T}.config_indicators").toPandas(), ["cliente", "grupo", "indicador"])

px = spark.table(f"{T}.silver_prices_daily").toPandas()
macro = spark.table(f"{T}.silver_macro").toPandas()
assets = spark.table(f"{T}.silver_assets").toPandas().set_index("asset_id")
clients = spark.table(f"{T}.silver_clients").toPandas().set_index("client_id")
stats = spark.table(f"{T}.gold_asset_stats").toPandas().set_index("asset_id")
wts = spark.table(f"{T}.gold_client_weights").toPandas()

px["date"] = pd.to_datetime(px.date)
px["volume"] = pd.to_numeric(px.volume, errors="coerce")
macro["date"] = pd.to_datetime(macro.date)
m = macro.set_index("date").sort_index()
close = px.pivot(index="date", columns="asset_id", values="close").sort_index()
ret = close.pct_change(fill_method=None).dropna(how="all")
adv = px.assign(usd=px.close * px.volume).groupby("asset_id").usd.mean()


def param(cid, regla):
    r = lim[(lim.regla == regla) & lim.cliente.isin([cid, "*"])]
    r = r[r.cliente == cid] if (r.cliente == cid).any() else r
    return None if r.empty else float(r.valor.iloc[0])


# COMMAND ----------

# MAGIC %md ## Indicadores
# MAGIC Todos en la moneda base del cliente: retorno diario en USD combinado con el cambio de `usd_<moneda>`.

# COMMAND ----------

def fx_ret(ccy):
    if ccy == "USD":
        return pd.Series(0.0, index=ret.index)
    col = f"usd_{ccy.lower()}"
    if col not in m:
        raise ValueError(f"No hay tipo de cambio {col} en silver_macro")
    return m[col].pct_change().reindex(ret.index).fillna(0.0)


def max_dd(r):
    cum = (1 + r.fillna(0)).cumprod()
    return float((cum / cum.cummax() - 1).min())


def ventana(r, n):
    return ((1 + r.fillna(0)).rolling(n).apply(np.prod, raw=True) - 1).dropna()


def indicadores(w_total, w_tramo, ccy, n):
    f = fx_ret(ccy)
    r_usd = (ret * w_total).sum(axis=1)
    r_b = (1 + r_usd) * (1 + f) - 1
    r_tramo = (1 + (ret * w_tramo).sum(axis=1)) * (1 + f) - 1
    vol_b, vol_usd = r_b.std() * np.sqrt(TD), r_usd.std() * np.sqrt(TD)
    return {
        "dd_base": max_dd(r_b),
        "vol_base": vol_b,
        "p05_6m_base": float(ventana(r_tramo, n).quantile(0.05)),
        "ret_anual_base": r_b.mean() * TD,
        "razon_vol_base_usd": vol_b / vol_usd,
    }


def puntaje(v, a0, a100):
    return float(np.clip((v - a0) / (a100 - a0), 0, 1) * 100)


# COMMAND ----------

# MAGIC %md ## Carteras candidatas
# MAGIC `actual` es la cartera de `gold_client_weights`. Si el cliente tiene `NECESIDAD_6M`, se añade `dividida`: (1 − n) de la cartera actual + n en un
# MAGIC tramo de liquidez con los 3 activos de menor volatilidad, drawdown y mayor liquidez combinados.

# COMMAND ----------

def tramo_liquidez(cols):
    rank = pd.DataFrame({
        "vol": stats.vol_anual.rank(),
        "dd": stats.max_drawdown.abs().rank(),
        "liq": adv.rank(ascending=False),
    }).sum(axis=1).sort_values()
    ids = list(rank.index[:3])
    inv = 1 / stats.vol_anual[ids]
    return (inv / inv.sum()).reindex(cols).fillna(0.0)


LIMITES = {  # regla -> (indicador que mide, cumple(valor, límite))
    "DD_MAX": ("dd_base", lambda v, l: v >= -l),
    "PESO_MAX_ALTA_VOL": ("peso_alta_vol", lambda v, l: v <= l),
    "P05_6M_MIN": ("p05_6m_base", lambda v, l: v >= -l),
}
DESC = lim.set_index(["cliente", "regla"]).descripcion.to_dict()

detail, checks, results, pesos_out = [], [], [], []
for cid in sorted(pesos.cliente.unique()):
    ccy = clients.loc[cid, "base_currency"]
    n = int(param(cid, "VENTANA_SESIONES"))
    w_act = wts[wts.cliente == cid].set_index("asset_id").peso.reindex(close.columns).fillna(0.0)
    w_act = w_act / w_act.sum()
    cands = {"actual": (w_act, w_act)}  # sin tramo aparte, el dinero se obtiene vendiendo a prorrata
    need = param(cid, "NECESIDAD_6M")
    if need:
        w_sl = tramo_liquidez(close.columns)
        cands["dividida"] = ((1 - need) * w_act + need * w_sl, w_sl)

    reglas_c = lim[lim.cliente.isin([cid, "*"]) & lim.regla.isin(LIMITES)]
    ver = pd.concat([lim[lim.cliente == cid], pesos[pesos.cliente == cid], ind[ind.cliente == cid]]).sort_values("vigente_desde").iloc[-1].twist_id

    for nombre, (w_tot, w_tr) in cands.items():
        v = indicadores(w_tot, w_tr, ccy, n)
        v["peso_alta_vol"] = float(w_tot[stats.alta_vol.reindex(w_tot.index).fillna(False)].sum())

        fallos = []
        for r in reglas_c.itertuples():
            nom, ok = LIMITES[r.regla]
            cumple = bool(ok(v[nom], float(r.valor)))
            checks.append({"cliente": cid, "cartera": nombre, "regla": r.regla, "valor": v[nom], "limite": float(r.valor),
                           "cumple": cumple, "descripcion": DESC.get((r.cliente, r.regla))})
            if not cumple:
                fallos.append(r.regla)

        gs = {}
        for g, sub in ind[ind.cliente == cid].groupby("grupo"):
            sc = []
            for i in sub.itertuples():
                s = puntaje(v[i.indicador], i.ancla_0, i.ancla_100)
                sc.append(s)
                detail.append({"cliente": cid, "cartera": nombre, "grupo": g, "indicador": i.indicador, "valor": float(v[i.indicador]),
                               "ancla_0": i.ancla_0, "ancla_100": i.ancla_100, "score": s})
            gs[g] = float(np.mean(sc))
        total = sum(pesos[(pesos.cliente == cid) & (pesos.grupo == g)].peso.iloc[0] * s for g, s in gs.items())
        results.append({"cliente": cid, "cartera": nombre, "apta": not fallos, "limites_incumplidos": ", ".join(fallos) or None,
                        **{f"score_{g}": s for g, s in gs.items()}, "score_total": None if fallos else float(total), "config_twist": ver})
        pesos_out += [{"cliente": cid, "cartera": nombre, "asset_id": a, "name": assets.name[a], "peso": float(p), "peso_tramo_liquidez": float(w_tr[a])}
                      for a, p in w_tot.items() if p > 1e-6]

res, det, chk, pw = map(pd.DataFrame, (results, detail, checks, pesos_out))
det = det.merge(pesos[["cliente", "grupo", "peso"]].rename(columns={"peso": "peso_grupo"}), on=["cliente", "grupo"])
display(res.round(3))

# COMMAND ----------

def save(df, name):
    sdf = spark.createDataFrame(df.astype(object).where(df.notna(), None))
    for f in sdf.schema.fields:  # una columna toda nula (p. ej. score_total si nada es apta) llega como void
        if f.dataType.simpleString() == "void":
            sdf = sdf.withColumn(f.name, sdf[f.name].cast("double"))
    sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{T}.{name}")
    print(f"{T}.{name}: {sdf.count()} filas")


save(res, "gold_score_results")
save(det, "gold_score_detail")
save(chk, "gold_limit_checks")
save(pw, "gold_portfolio_weights")
