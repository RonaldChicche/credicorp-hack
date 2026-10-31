# Databricks notebook source
# MAGIC %md
# MAGIC # Análisis de portafolio
# MAGIC Usa las tablas `*_clean` generadas por `explorar_datos`. Cubre riesgo/retorno por activo,
# MAGIC correlaciones, sensibilidad macro, régimen de riesgo, fundamentales, eventos y encaje con los perfiles de cliente.

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "credicorp_hack")
T = f"{dbutils.widgets.get('catalog')}.{dbutils.widgets.get('schema')}"

px = spark.table(f"{T}.prices_daily_clean").toPandas()
macro = spark.table(f"{T}.macro_clean").toPandas()
fund = spark.table(f"{T}.fundamentals_clean").toPandas()
assets = spark.table(f"{T}.assets").toPandas()
clients = spark.table(f"{T}.clients").toPandas()
events = spark.table(f"{T}.events").toPandas()

px["date"] = pd.to_datetime(px.date)
macro["date"] = pd.to_datetime(macro.date)
close = px.pivot(index="date", columns="asset_id", values="close").sort_index()
ret = close.pct_change(fill_method=None).dropna(how="all")
names = assets.set_index("asset_id").name
TRADING_DAYS = 252
print(f"{close.shape[1]} activos, {close.index.min().date()} -> {close.index.max().date()}, {len(close)} días")

# COMMAND ----------

# MAGIC %md ## 1. Retorno, volatilidad y drawdown por activo

# COMMAND ----------

def max_drawdown(series):
    cum = (1 + series.fillna(0)).cumprod()
    return float((cum / cum.cummax() - 1).min())


rf = macro.us_10y_yield.mean() / 100
stats = pd.DataFrame({
    "retorno_total": close.iloc[-1] / close.iloc[0] - 1,
    "retorno_anual": ret.mean() * TRADING_DAYS,
    "vol_anual": ret.std() * np.sqrt(TRADING_DAYS),
    "max_drawdown": ret.apply(max_drawdown),
})
stats["sharpe"] = (stats.retorno_anual - rf) / stats.vol_anual
stats = assets.set_index("asset_id")[["name", "sector", "country"]].join(stats).sort_values("sharpe", ascending=False)
display(stats.round(3).reset_index())

# COMMAND ----------

fig, ax = plt.subplots(figsize=(9, 4.5))
(close / close.iloc[0]).rename(columns=names).plot(ax=ax, lw=1.2)
ax.set_title("Precio rebasado a 1.0")
ax.legend(ncol=2, fontsize=8)
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md ## 2. Correlaciones

# COMMAND ----------

corr = ret.corr()
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im)
ax.set_title("Correlación de retornos diarios")
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md ## 3. Sensibilidad macro
# MAGIC Beta al `market_factor` y correlación con variaciones de tasas y monedas.

# COMMAND ----------

m = macro.set_index("date").sort_index()
mchg = pd.DataFrame({
    "market_factor": m.market_factor,
    "d_us10y": m.us_10y_yield.diff(),
    "cop": m.usd_cop.pct_change(),
    "pen": m.usd_pen.pct_change(),
    "mxn": m.usd_mxn.pct_change(),
    "clp": m.usd_clp.pct_change(),
})
joined = ret.join(mchg, how="inner")
beta = {a: joined[a].cov(joined.market_factor) / joined.market_factor.var() for a in ret.columns}
sens = pd.DataFrame({"beta_mercado": beta})
for c in ["d_us10y", "cop", "pen", "mxn", "clp"]:
    sens[f"corr_{c}"] = [joined[a].corr(joined[c]) for a in ret.columns]
display(sens.join(names).round(3).reset_index())

# COMMAND ----------

# MAGIC %md ## 4. Régimen de riesgo
# MAGIC Portafolio igualmente ponderado según el régimen (`Normal`, `Stress`, `Recovery`).

# COMMAND ----------

ew = ret.mean(axis=1).rename("ret_ew")
reg = ew.to_frame().join(m.risk_regime, how="inner")
by_reg = reg.groupby("risk_regime").ret_ew.agg(dias="count", retorno_medio_diario="mean", vol_diaria="std")
by_reg["retorno_medio_anual"] = by_reg.retorno_medio_diario * TRADING_DAYS
display(by_reg.round(5).reset_index())

per_asset_reg = ret.join(m.risk_regime, how="inner").groupby("risk_regime").mean() * TRADING_DAYS
display(per_asset_reg.round(3).reset_index())

# COMMAND ----------

# MAGIC %md ## 5. Fundamentales
# MAGIC Último trimestre por emisor: margen EBITDA, deuda neta / EBITDA anualizado y crecimiento de ingresos a 4 trimestres.

# COMMAND ----------

f = fund.sort_values(["issuer_id", "period_end"]).copy()
f["rev_yoy"] = f.groupby("issuer_id").revenue_usd_m.pct_change(4)
last = f.groupby("issuer_id").tail(1).copy()
last["margen_ebitda"] = last.ebitda_usd_m / last.revenue_usd_m
last["deuda_neta"] = last.debt_usd_m - last.cash_usd_m
last["deuda_neta_ebitda"] = last.deuda_neta / (last.ebitda_usd_m * 4)
fund_view = assets[["asset_id", "issuer_id", "name", "sector"]].merge(
    last[["issuer_id", "period_end", "margen_ebitda", "deuda_neta_ebitda", "rev_yoy"]], on="issuer_id")
display(fund_view.round(3).sort_values("deuda_neta_ebitda", ascending=False))

# COMMAND ----------

# MAGIC %md ## 6. Impacto de eventos
# MAGIC Retorno acumulado desde el cierre previo hasta 3 sesiones después del evento, frente al portafolio igualmente ponderado.
# MAGIC El `scope` puede ser un activo o un sector.

# COMMAND ----------

sector_assets = assets.groupby("sector").asset_id.apply(list).to_dict()
rows = []
for _, e in events.iterrows():
    targets = [e.scope] if e.scope in close.columns else sector_assets.get(e.scope, [])
    d0 = pd.Timestamp(e.event_date)
    idx = close.index
    i = idx.searchsorted(d0)
    if i == 0 or i + 3 >= len(idx):
        continue
    window = slice(idx[i - 1], idx[i + 3])
    cum = close.loc[window].iloc[-1] / close.loc[window].iloc[0] - 1
    rows.append({
        "fecha": d0.date(), "scope": e.scope, "tipo": e.event_type, "severidad": e.severity,
        "retorno_scope": cum[targets].mean(), "retorno_mercado_ew": cum.mean(),
        "exceso": cum[targets].mean() - cum.mean(), "titular": e.headline,
    })
display(pd.DataFrame(rows).round(4))

# COMMAND ----------

# MAGIC %md ## 7. Encaje con perfiles de cliente
# MAGIC Se arma una cartera por cliente con ponderación inversa a la volatilidad sobre los activos elegibles, aplicando la
# MAGIC restricción de cada perfil. Es un punto de partida, no una recomendación de inversión.

# COMMAND ----------

meta = assets.set_index("asset_id")
vol = stats.vol_anual
high_vol = set(vol[vol >= vol.quantile(2 / 3)].index)  # tercil superior = "alta volatilidad"


def inv_vol_weights(ids):
    w = 1 / vol[ids]
    return w / w.sum()


def cap_group(w, groups, cap, capped=None):
    """Limita el peso de los grupos en `capped` (todos si es None) reasignando el exceso al resto (iterativo)."""
    w = w.copy()
    for _ in range(50):
        g = w.groupby(groups.reindex(w.index)).sum()
        over = g[(g > cap + 1e-9) & (g.index.isin(capped) if capped is not None else True)]
        if over.empty:
            break
        for grp in over.index:
            members = groups.reindex(w.index) == grp
            w[members] *= cap / g[grp]
        free = w.index[~groups.reindex(w.index).isin(over.index)]
        if len(free) == 0:
            break
        w[free] += (1 - w.sum()) * w[free] / w[free].sum()
    return w


def build(client):
    ids = list(close.columns)
    tol = client.max_drawdown_tolerance_pct / 100
    if client.risk_tolerance == "Media-Baja":
        ids = [a for a in ids if a not in high_vol]
    w = inv_vol_weights(ids)
    c = client.constraint
    if "un solo sector" in c:
        w = cap_group(w, meta.sector, 0.45)
    elif "alta volatilidad" in c:
        hv = pd.Series({a: "alta" if a in high_vol else "resto" for a in w.index})
        w = cap_group(w, hv, 0.25, capped=["alta"])
    assert abs(w.sum() - 1) < 1e-6, f"{client.client_id}: los pesos suman {w.sum():.3f}"
    port = (ret[w.index] * w).sum(axis=1)
    dd = max_drawdown(port)
    ok = {
        "drawdown_ok": abs(dd) <= tol,
        "max_sector": w.groupby(meta.sector.reindex(w.index)).sum().max(),
        "n_paises": meta.country.reindex(w.index)[w > 0.01].nunique(),
        "n_sectores": meta.sector.reindex(w.index)[w > 0.01].nunique(),
        "peso_alta_vol": w[w.index.isin(high_vol)].sum(),
    }
    return w, port, dd, ok


summary = []
weights_rows = []
fig, ax = plt.subplots(figsize=(9, 4))
for _, cl in clients.iterrows():
    w, port, dd, ok = build(cl)
    summary.append({
        "cliente": cl.client_id, "perfil": cl.profile_name, "restriccion": cl.constraint,
        "retorno_anual": port.mean() * TRADING_DAYS, "vol_anual": port.std() * np.sqrt(TRADING_DAYS),
        "max_drawdown": dd, "tolerancia_dd": -cl.max_drawdown_tolerance_pct / 100, **ok,
    })
    weights_rows += [{"cliente": cl.client_id, "asset_id": a, "peso": float(x)} for a, x in w.items() if x > 0.001]
    (1 + port.fillna(0)).cumprod().plot(ax=ax, label=cl.profile_name)
    print(f"\n{cl.client_id} {cl.profile_name}")
    print(w[w > 0.001].round(3).rename(names).sort_values(ascending=False).to_string())
ax.legend()
ax.set_title("Valor acumulado de las carteras sugeridas")
plt.tight_layout()
display(fig)
display(pd.DataFrame(summary).round(3))

# COMMAND ----------

# MAGIC %md ## 8. Guardar resultados

# COMMAND ----------

def save(df, name):
    spark.createDataFrame(df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{T}.{name}")


save(stats.reset_index(), "analysis_asset_stats")
save(pd.DataFrame(summary), "analysis_client_summary")
save(pd.DataFrame(weights_rows), "analysis_client_weights")
