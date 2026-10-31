# Databricks notebook source
# MAGIC %md
# MAGIC # Simulación de portafolios
# MAGIC Busca qué carteras se le pueden recomendar a un cliente, con las mismas reglas de `config_*` que usa `01_calcular_score`.
# MAGIC
# MAGIC **Método**
# MAGIC 1. **Muestreo:** se generan miles de carteras al azar. Cada una son dos vectores de pesos sobre los activos: el **núcleo** y el **tramo de liquidez**
# MAGIC    (la fracción `NECESIDAD_6M` del portafolio). Se evalúan con los mismos indicadores, en la moneda base del cliente.
# MAGIC 2. **Refinamiento:** se perturban las mejores y se conservan las que mejoran.
# MAGIC 3. **Orden:** primero las que cumplen todos los límites duros, por `score_total`; si no hay suficientes, las de menor violación de límites.
# MAGIC
# MAGIC Es una búsqueda aleatoria: encuentra buenas carteras, pero no prueba que sean el óptimo ni que no exista una mejor.
# MAGIC Salidas: `gold_sim_candidates`, `gold_sim_weights` y `gold_sim_summary`.

# COMMAND ----------

import numpy as np
import pandas as pd

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "credicorp_hack")
dbutils.widgets.text("cliente", "CL_02")
dbutils.widgets.text("n_sim", "100000")
dbutils.widgets.text("rondas_refinamiento", "15")
dbutils.widgets.text("top_n", "50")
dbutils.widgets.text("seed", "42")
T = f"{dbutils.widgets.get('catalog')}.{dbutils.widgets.get('schema')}"
CID = dbutils.widgets.get("cliente")
N_SIM, RONDAS, TOP_N, SEED = (int(dbutils.widgets.get(k)) for k in ("n_sim", "rondas_refinamiento", "top_n", "seed"))
TD, CHUNK, POOL, HIJOS, SIGMA = 252, 10_000, 200, 10, 0.04
rng = np.random.default_rng(SEED)


def vigente(df, keys):
    return df.sort_values("vigente_desde").drop_duplicates(keys, keep="last")


lim = vigente(spark.table(f"{T}.config_limits").toPandas(), ["cliente", "regla"])
pesos = vigente(spark.table(f"{T}.config_weights").toPandas(), ["cliente", "grupo"])
ind = vigente(spark.table(f"{T}.config_indicators").toPandas(), ["cliente", "grupo", "indicador"])


def param(regla):
    r = lim[(lim.regla == regla) & lim.cliente.isin([CID, "*"])]
    r = r[r.cliente == CID] if (r.cliente == CID).any() else r
    return None if r.empty else float(r.valor.iloc[0])


DD, PAV, P05, NEED, NV = (param(k) for k in ("DD_MAX", "PESO_MAX_ALTA_VOL", "P05_6M_MIN", "NECESIDAD_6M", "VENTANA_SESIONES"))
assert None not in (DD, PAV, P05, NEED, NV), f"{CID} no tiene todos los límites configurados en config_limits"
NV = int(NV)
ind_c = ind[ind.cliente == CID]
ANCLAS = {r.indicador: (r.ancla_0, r.ancla_100) for r in ind_c.itertuples()}
GRUPOS = ind_c.groupby("grupo").indicador.apply(list).to_dict()
PESO = pesos[pesos.cliente == CID].set_index("grupo").peso.to_dict()
print(f"{CID}: DD ≤ {DD:.0%}, alta vol ≤ {PAV:.0%}, p05 6m ≥ -{P05:.0%}, tramo de liquidez {NEED:.0%}")

# COMMAND ----------

# MAGIC %md ## Datos y evaluación vectorizada
# MAGIC Las fórmulas son las de `01_calcular_score`, aplicadas a muchas carteras a la vez. Se comprueba contra `gold_score_detail`.

# COMMAND ----------

px = spark.table(f"{T}.silver_prices_daily").toPandas()
macro = spark.table(f"{T}.silver_macro").toPandas()
assets = spark.table(f"{T}.silver_assets").toPandas().set_index("asset_id")
clients = spark.table(f"{T}.silver_clients").toPandas().set_index("client_id")
stats = spark.table(f"{T}.gold_asset_stats").toPandas().set_index("asset_id")

px["date"] = pd.to_datetime(px.date)
macro["date"] = pd.to_datetime(macro.date)
close = px.pivot(index="date", columns="asset_id", values="close").sort_index()
ret = close.pct_change(fill_method=None).dropna(how="all")
COLS = list(close.columns)
A = len(COLS)
R = ret.fillna(0.0).to_numpy()
ALTA = stats.alta_vol.reindex(COLS).fillna(False).to_numpy(dtype=float)

ccy = clients.loc[CID, "base_currency"]
if ccy == "USD":
    FX = np.zeros(len(ret))
else:
    FX = macro.set_index("date").sort_index()[f"usd_{ccy.lower()}"].pct_change().reindex(ret.index).fillna(0.0).to_numpy()


def evaluar(Wc, Wt):
    """Wc, Wt: (n × activos). Devuelve pesos totales, indicadores, scores por grupo, total y violación de límites."""
    Wtot = (1 - NEED) * Wc + NEED * Wt
    r_usd = R @ Wtot.T
    r_b = (1 + r_usd) * (1 + FX[:, None]) - 1
    cum = np.cumprod(1 + r_b, axis=0)
    vol_b = r_b.std(axis=0, ddof=1) * np.sqrt(TD)
    vol_u = r_usd.std(axis=0, ddof=1) * np.sqrt(TD)
    r_t = (1 + R @ Wt.T) * (1 + FX[:, None]) - 1
    cp = np.vstack([np.ones((1, r_t.shape[1])), np.cumprod(1 + r_t, axis=0)])
    v = {
        "dd_base": (cum / np.maximum.accumulate(cum, axis=0) - 1).min(axis=0),
        "vol_base": vol_b,
        "p05_6m_base": np.quantile(cp[NV:] / cp[:-NV] - 1, 0.05, axis=0),
        "ret_anual_base": r_b.mean(axis=0) * TD,
        "razon_vol_base_usd": vol_b / vol_u,
        "peso_alta_vol": Wtot @ ALTA,
    }
    sc = {k: np.clip((v[k] - a0) / (a1 - a0), 0, 1) * 100 for k, (a0, a1) in ANCLAS.items()}
    grp = {g: np.mean([sc[i] for i in inds], axis=0) for g, inds in GRUPOS.items()}
    total = sum(PESO[g] * grp[g] for g in grp)
    viol = (np.maximum(0, -v["dd_base"] - DD) / DD + np.maximum(0, v["peso_alta_vol"] - PAV) / PAV
            + np.maximum(0, -v["p05_6m_base"] - P05) / P05)
    return {"Wc": Wc, "Wt": Wt, "Wtot": Wtot, "v": v, "grp": grp, "total": total, "viol": viol}


def juntar(a, b):
    return {"Wc": np.vstack([a["Wc"], b["Wc"]]), "Wt": np.vstack([a["Wt"], b["Wt"]]), "Wtot": np.vstack([a["Wtot"], b["Wtot"]]),
            "v": {k: np.concatenate([a["v"][k], b["v"][k]]) for k in a["v"]},
            "grp": {k: np.concatenate([a["grp"][k], b["grp"][k]]) for k in a["grp"]},
            "total": np.concatenate([a["total"], b["total"]]), "viol": np.concatenate([a["viol"], b["viol"]])}


def tomar(e, idx):
    return {"Wc": e["Wc"][idx], "Wt": e["Wt"][idx], "Wtot": e["Wtot"][idx], "v": {k: x[idx] for k, x in e["v"].items()},
            "grp": {k: x[idx] for k, x in e["grp"].items()}, "total": e["total"][idx], "viol": e["viol"][idx]}


def mejores(e, k):
    """Primero las que cumplen todo (violación 0) por score descendente; luego las de menor violación."""
    return tomar(e, np.lexsort((-e["total"], e["viol"]))[:k])


# Comprobación: las carteras base de 01_calcular_score deben dar los mismos indicadores
base_w = spark.table(f"{T}.gold_portfolio_weights").filter(f"cliente = '{CID}'").toPandas()
base_d = spark.table(f"{T}.gold_score_detail").filter(f"cliente = '{CID}'").toPandas()
for nombre, g in base_w.groupby("cartera"):
    wt = g.set_index("asset_id").peso.reindex(COLS).fillna(0.0).to_numpy()
    wtr = g.set_index("asset_id").peso_tramo_liquidez.reindex(COLS).fillna(0.0).to_numpy()
    # evaluar() recompone el total como (1-n)·núcleo + n·tramo; aquí el total ya viene dado, así que se pasa el núcleo equivalente
    nucleo = (wt - NEED * wtr) / (1 - NEED)
    e = evaluar(nucleo[None, :], wtr[None, :])
    esperado = base_d[base_d.cartera == nombre].set_index("indicador").valor
    for k in ANCLAS:
        assert abs(e["v"][k][0] - esperado[k]) < 1e-9, f"{nombre}.{k}: {e['v'][k][0]} != {esperado[k]}"
print("Evaluación vectorizada coincide con gold_score_detail")

# COMMAND ----------

# MAGIC %md ## Muestreo y refinamiento

# COMMAND ----------

def dirichlet(n):
    alpha = rng.choice([0.3, 1.0, 3.0], size=n)  # mezcla de carteras concentradas y diversificadas
    g = rng.gamma(alpha[:, None], size=(n, A)) + 1e-12
    return g / g.sum(axis=1, keepdims=True)


agg = {"n": 0, "aptas": 0, "dd_mejor": -np.inf}


def registrar(e):
    agg["n"] += len(e["viol"])
    agg["aptas"] += int((e["viol"] == 0).sum())
    agg["dd_mejor"] = max(agg["dd_mejor"], float(e["v"]["dd_base"].max()))


pool = None
for ini in range(0, N_SIM, CHUNK):
    n = min(CHUNK, N_SIM - ini)
    e = evaluar(dirichlet(n), dirichlet(n))
    registrar(e)
    pool = mejores(e if pool is None else juntar(pool, e), POOL)
print(f"Muestreo: {agg['n']:,} carteras, {agg['aptas']:,} cumplen todos los límites")

for r in range(RONDAS):
    k = len(pool["viol"])
    padre_c, padre_t = np.repeat(pool["Wc"], HIJOS, axis=0), np.repeat(pool["Wt"], HIJOS, axis=0)

    def mutar(W):
        h = np.clip(W + rng.normal(0, SIGMA, W.shape), 0, None)
        s = h.sum(axis=1, keepdims=True)
        return np.where(s > 0, h / np.where(s > 0, s, 1), W)

    hijos = evaluar(mutar(padre_c), mutar(padre_t))
    registrar(hijos)
    pool = mejores(juntar(pool, hijos), POOL)
print(f"Tras refinar: {agg['n']:,} evaluadas, {agg['aptas']:,} aptas. Mejor drawdown encontrado: {agg['dd_mejor']:.2%} (límite -{DD:.0%})")

# COMMAND ----------

# MAGIC %md ## Candidatas finales

# COMMAND ----------

top = mejores(pool, TOP_N)
nombres = assets.name.reindex(COLS).tolist()
apta = top["viol"] == 0


def incumplidos(i):
    f = []
    if -top["v"]["dd_base"][i] > DD:
        f.append("DD_MAX")
    if top["v"]["peso_alta_vol"][i] > PAV:
        f.append("PESO_MAX_ALTA_VOL")
    if -top["v"]["p05_6m_base"][i] > P05:
        f.append("P05_6M_MIN")
    return ", ".join(f) or None


cand = pd.DataFrame({
    "cliente": CID, "sim_id": np.arange(1, len(apta) + 1), "apta": apta, "violacion": top["viol"],
    "limites_incumplidos": [incumplidos(i) for i in range(len(apta))],
    **{k: top["v"][k] for k in top["v"]},
    **{f"score_{g}": s for g, s in top["grp"].items()},
    "score_total": np.where(apta, top["total"], np.nan),
})
pesos_df = pd.DataFrame([
    {"cliente": CID, "sim_id": i + 1, "asset_id": a, "name": nombres[j],
     "peso_total": float(top["Wtot"][i, j]), "peso_nucleo": float(top["Wc"][i, j]), "peso_tramo_liquidez": float(top["Wt"][i, j])}
    for i in range(len(apta)) for j, a in enumerate(COLS) if top["Wtot"][i, j] > 1e-4
])
resumen = pd.DataFrame([{
    "cliente": CID, "n_simuladas": agg["n"], "n_aptas": agg["aptas"], "pct_aptas": agg["aptas"] / agg["n"],
    "mejor_drawdown": agg["dd_mejor"], "limite_drawdown": -DD, "mejor_score_total": float(cand.score_total.max()) if apta.any() else None,
    "semilla": SEED, "rondas_refinamiento": RONDAS,
}])
display(resumen)
display(cand.head(10).round(4))

# COMMAND ----------

def save(df, name):
    sdf = spark.createDataFrame(df.astype(object).where(df.notna(), None))
    for f in sdf.schema.fields:
        if f.dataType.simpleString() == "void":
            sdf = sdf.withColumn(f.name, sdf[f.name].cast("double"))
    sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{T}.{name}")
    print(f"{T}.{name}: {sdf.count()} filas")


save(cand, "gold_sim_candidates")
save(pesos_df, "gold_sim_weights")
save(resumen, "gold_sim_summary")
