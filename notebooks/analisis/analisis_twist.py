# Databricks notebook source
# MAGIC %md
# MAGIC # Análisis del twist (17-ago-2026)
# MAGIC Tres noticias posteriores al corte de los datos (14-ago-2026) obligan a revisar las recomendaciones:
# MAGIC
# MAGIC | id | Qué cambia | Qué pide el twist |
# MAGIC |---|---|---|
# MAGIC | TW_01 | `us_10y_yield` +65 pb en una semana | ¿Cambia la lectura de sensibilidad a tasas / valoración? |
# MAGIC | TW_02 | CL_02 necesitará 30% del portafolio en 6 meses | ¿Sigue siendo adecuada la recomendación? |
# MAGIC | TW_03 | COP se deprecia 8% frente al USD | Impacto para un cliente con moneda base COP (CL_02) |
# MAGIC
# MAGIC **Límites del dato:** solo hay ~1 año de histórico diario (2025-08 a 2026-08). Los choques del twist están fuera de lo observado,
# MAGIC así que todo es un *escenario por extrapolación lineal*, no una predicción. El histórico intradía (desde 2020) no se usa:
# MAGIC sus cierres difieren de los diarios en promedio ~14%, así que no es la misma serie.

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "credicorp_hack")
dbutils.widgets.text("participation", "0.10")
dbutils.widgets.text("liquidation_days", "5")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
T = f"{catalog}.{schema}"
PARTICIPATION = float(dbutils.widgets.get("participation"))  # % del volumen diario que se puede vender sin mover el precio
LIQ_DAYS = int(dbutils.widgets.get("liquidation_days"))

twist = pd.read_csv(f"/Volumes/{catalog}/{schema}/raw/dataset_twist.csv")
display(twist)

px = spark.table(f"{T}.prices_daily_clean").toPandas()
macro = spark.table(f"{T}.macro_clean").toPandas()
assets = spark.table(f"{T}.assets").toPandas()
fund = spark.table(f"{T}.fundamentals_clean").toPandas()
weights = spark.table(f"{T}.analysis_client_weights").toPandas()
stats = spark.table(f"{T}.analysis_asset_stats").toPandas().set_index("asset_id")

px["date"] = pd.to_datetime(px.date)
px["volume"] = pd.to_numeric(px.volume, errors="coerce")
macro["date"] = pd.to_datetime(macro.date)
m = macro.set_index("date").sort_index()
close = px.pivot(index="date", columns="asset_id", values="close").sort_index()
ret = close.pct_change(fill_method=None).dropna(how="all")
names = assets.set_index("asset_id").name
TD = 252


def client_w(cid):
    w = weights[weights.cliente == cid].set_index("asset_id").peso
    return (w / w.sum()).reindex(close.columns).fillna(0.0)


def port_ret(w):
    return (ret * w).sum(axis=1)


def max_dd(r):
    cum = (1 + r.fillna(0)).cumprod()
    return float((cum / cum.cummax() - 1).min())


def ols(y, X):
    """OLS con constante; devuelve coeficientes y t-stats (sin statsmodels)."""
    X1 = np.column_stack([np.ones(len(X)), np.asarray(X)])
    b, *_ = np.linalg.lstsq(X1, np.asarray(y), rcond=None)
    resid = np.asarray(y) - X1 @ b
    s2 = resid @ resid / (len(y) - X1.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X1.T @ X1)))
    return b, b / se


# COMMAND ----------

# MAGIC %md ## TW_01 — Sube la tasa a 10 años +65 pb

# COMMAND ----------

SHOCK_BP = 65
d_bp = (m.us_10y_yield.diff() * 100).rename("dbp")
week_bp = m.us_10y_yield.diff(5) * 100
print(f"Rango observado del 10y: {m.us_10y_yield.min():.2f}% a {m.us_10y_yield.max():.2f}%")
print(f"Mayor variación semanal observada: {week_bp.abs().max():.1f} pb  ->  el choque de {SHOCK_BP} pb es {SHOCK_BP / week_bp.abs().max():.1f}x lo visto")
print(f"Variación diaria típica: {d_bp.std():.1f} pb")

rows = []
for a in ret.columns:
    d = pd.concat([ret[a].rename("r"), d_bp, m.market_factor], axis=1).dropna()
    b_tot, t_tot = ols(d.r, d[["dbp"]])
    b_dir, t_dir = ols(d.r, d[["dbp", "market_factor"]])
    rows.append({
        "asset_id": a, "name": names[a],
        "beta_total_por_pb": b_tot[1], "t_total": t_tot[1],
        "beta_directa_por_pb": b_dir[1], "t_directa": t_dir[1],
        "impacto_65pb_total": b_tot[1] * SHOCK_BP,
        "impacto_65pb_directo": b_dir[1] * SHOCK_BP,
    })
tw1 = pd.DataFrame(rows).set_index("asset_id")

# apalancamiento (deuda neta / EBITDA anualizado) como lente de valoración
f = fund.sort_values(["issuer_id", "period_end"]).groupby("issuer_id").tail(1).set_index("issuer_id")
lev = ((f.debt_usd_m - f.cash_usd_m) / (f.ebitda_usd_m * 4)).rename("deuda_neta_ebitda")
tw1 = tw1.join(assets.set_index("asset_id").issuer_id.map(lev).rename("deuda_neta_ebitda")).join(assets.set_index("asset_id").sector)
tw1["lectura"] = np.where(tw1.t_total.abs() >= 2, np.where(tw1.beta_total_por_pb < 0, "Sensible: pierde con alza de tasas", "Sensible: gana con alza de tasas"),
                          "No concluyente (|t| < 2)")
display(tw1.round(4).reset_index())

# COMMAND ----------

fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#c0392b" if v < 0 else "#27ae60" for v in tw1.impacto_65pb_total]
ax.barh(tw1.name, tw1.impacto_65pb_total * 100, color=colors)
ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("Impacto estimado en el precio (%) ante +65 pb")
ax.set_title("TW_01: escenario lineal de tasas (extrapolado)")
plt.tight_layout()
display(fig)

# COMMAND ----------

corr_lev = tw1[["beta_total_por_pb", "deuda_neta_ebitda"]].corr().iloc[0, 1]
print(f"Correlación sensibilidad a tasas vs. apalancamiento (n=8, muy poco poder estadístico): {corr_lev:.2f}")

imp_clients = []
for cid in ["CL_01", "CL_02", "CL_03"]:
    w = client_w(cid)
    imp_clients.append({"cliente": cid, "impacto_total_65pb": float((w * tw1.impacto_65pb_total).sum()),
                        "impacto_directo_65pb": float((w * tw1.impacto_65pb_directo).sum())})
tw1_clients = pd.DataFrame(imp_clients)
display(tw1_clients.round(4))

# COMMAND ----------

# MAGIC %md ## TW_02 — CL_02 necesita 30% del portafolio en 6 meses
# MAGIC El perfil pasa de `liquidity_need = Media` y horizonte 36 meses a una porción (30%) con horizonte de 6 meses.
# MAGIC Se compara la cartera actual con una **cartera dividida**: 70% núcleo (la cartera previa de CL_02) + 30% *sleeve* de liquidez.
# MAGIC El sleeve toma los 3 activos con menor volatilidad, menor drawdown y mayor liquidez combinados (ranking por riesgo, no por retorno pasado).

# COMMAND ----------

NEED = 0.30
WINDOW = 126  # ~6 meses de sesiones
w_core = client_w("CL_02")

adv_usd = (px.assign(usd=px.close * px.volume).groupby("asset_id").usd.mean())

rank = pd.DataFrame({
    "vol": stats.vol_anual.rank(),
    "dd": stats.max_drawdown.abs().rank(),
    "liq": adv_usd.rank(ascending=False),
}).sum(axis=1).sort_values()
sleeve_ids = list(rank.index[:3])
inv = 1 / stats.vol_anual[sleeve_ids]
w_sleeve = (inv / inv.sum()).reindex(close.columns).fillna(0.0)
w_new = (1 - NEED) * w_core + NEED * w_sleeve
print("Sleeve de liquidez:", {names[a]: round(float(w_sleeve[a]), 3) for a in sleeve_ids})


def window_stats(r, label):
    roll = (1 + r.fillna(0)).rolling(WINDOW).apply(np.prod, raw=True).dropna() - 1
    return {
        "cartera": label, "vol_anual": r.std() * np.sqrt(TD), "max_drawdown": max_dd(r),
        "ret_6m_mediana": roll.median(), "ret_6m_p05": roll.quantile(0.05), "ret_6m_peor": roll.min(),
        "prob_perdida_6m": (roll < 0).mean(), "n_ventanas": len(roll),
    }


def capacity(sold_weights):
    """AUM máximo (USD) que puede vender `sold_weights` en LIQ_DAYS sesiones sin superar PARTICIPATION del volumen diario."""
    s = sold_weights[sold_weights > 1e-9]
    return float((PARTICIPATION * adv_usd[s.index] * LIQ_DAYS / s).min())


cmp2 = pd.DataFrame([
    window_stats(port_ret(w_core), "Actual (núcleo CL_02)"),
    window_stats(port_ret(w_new), "Dividida 70/30"),
    window_stats(port_ret(w_sleeve), "Solo sleeve 30%"),
])
cmp2["tolerancia_dd"] = -0.10
display(cmp2.round(4))

# COMMAND ----------

cap_now = capacity(NEED * w_core)  # vender 30% a prorrata de la cartera actual
cap_new = capacity(NEED * w_sleeve)  # vender solo el sleeve
print(f"AUM máximo para liquidar 30% en {LIQ_DAYS} sesiones (≤{PARTICIPATION:.0%} del volumen diario):")
print(f"  cartera actual : USD {cap_now / 1e6:,.0f} M")
print(f"  cartera dividida: USD {cap_new / 1e6:,.0f} M")
print("Liquidez diaria (USD M):", (adv_usd / 1e6).round(1).rename(names).to_dict())

# COMMAND ----------

roll_now = (1 + port_ret(w_core)).rolling(WINDOW).apply(np.prod, raw=True).dropna() - 1
roll_new = (1 + port_ret(w_new)).rolling(WINDOW).apply(np.prod, raw=True).dropna() - 1
roll_slv = (1 + port_ret(w_sleeve)).rolling(WINDOW).apply(np.prod, raw=True).dropna() - 1
fig, ax = plt.subplots(figsize=(9, 4))
for s, lab in [(roll_now, "Actual"), (roll_new, "Dividida 70/30"), (roll_slv, "Solo sleeve")]:
    (s * 100).plot(ax=ax, label=lab)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("Retorno a 6 meses (%)")
ax.set_title("TW_02: retorno móvil a 126 sesiones")
ax.legend()
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md ## TW_03 — El COP se deprecia 8% frente al USD
# MAGIC CL_02 tiene moneda base COP, pero todos los activos cotizan en USD: el cliente está **largo USD**. Una depreciación del COP
# MAGIC *sube* el valor de la cartera en pesos. "Deprecia 8%" admite dos lecturas: USD/COP × 1.08 (+8.0%) o el COP pierde 8% de valor (USD/COP × 1/0.92 = +8.7%).

# COMMAND ----------

fx = m.usd_cop
fx_ret = fx.pct_change().rename("fx")
week_fx = fx.pct_change(5)
print(f"USD/COP observado: {fx.min():,.0f} a {fx.max():,.0f}  (rango {fx.max() / fx.min() - 1:.1%})")
print(f"Mayor variación semanal observada: {week_fx.abs().max():.1%}; el choque de 8% es {0.08 / week_fx.abs().max():.1f}x lo visto")
print(f"Volatilidad anual USD/COP: {fx_ret.std() * np.sqrt(TD):.1%}")

SHOCK_LOW, SHOCK_HIGH = 0.08, 1 / 0.92 - 1


def cop_view(w, label):
    r_usd = port_ret(w)
    j = pd.concat([r_usd.rename("usd"), fx_ret], axis=1).dropna()
    r_cop = (1 + j.usd) * (1 + j.fx) - 1
    h = ols(r_cop, j[["fx"]])[0][1]  # cobertura de varianza mínima (proporción del portafolio a cubrir)
    hedged = r_cop - h * j.fx
    return {
        "cartera": label, "vol_usd": j.usd.std() * np.sqrt(TD), "vol_cop": r_cop.std() * np.sqrt(TD),
        "maxdd_usd": max_dd(j.usd), "maxdd_cop": max_dd(r_cop), "corr_cartera_fx": j.usd.corr(j.fx),
        "ratio_cobertura_min_var": h, "vol_cop_cubierta": hedged.std() * np.sqrt(TD),
        "efecto_traslacion_8pct": SHOCK_LOW, "efecto_traslacion_8_7pct": SHOCK_HIGH,
    }


tw3 = pd.DataFrame([cop_view(w_core, "Actual"), cop_view(w_new, "Dividida 70/30")])
display(tw3.round(4))

# COMMAND ----------

# MAGIC %md Efecto sobre el **30% que se necesitará en 6 meses** (en COP): ganancia por traslación si el COP no se revierte, y pérdida si el choque se revierte.

# COMMAND ----------

scen = pd.DataFrame({
    "escenario": ["COP se deprecia 8% (USD/COP +8.0%)", "COP se deprecia 8% (USD/COP +8.7%)", "Choque se revierte (USD/COP -7.4%)"],
    "efecto_sobre_valor_en_COP": [SHOCK_LOW, SHOCK_HIGH, 1 / (1 + SHOCK_LOW) - 1],
})
scen["efecto_sobre_el_30%_necesitado"] = scen.efecto_sobre_valor_en_COP * NEED
display(scen.round(4))

others = pd.DataFrame({
    "par": ["USD/PEN (CL_03)", "USD/MXN", "USD/CLP"],
    "corr_con_USD/COP": [m.usd_pen.pct_change().corr(fx_ret), m.usd_mxn.pct_change().corr(fx_ret), m.usd_clp.pct_change().corr(fx_ret)],
})
others["beta_a_USD/COP"] = [m[c].pct_change().cov(fx_ret) / fx_ret.var() for c in ["usd_pen", "usd_mxn", "usd_clp"]]
others["efecto_implicito_choque_8pct"] = others["beta_a_USD/COP"] * SHOCK_LOW
display(others.round(4))

# COMMAND ----------

# MAGIC %md ## Resumen de decisiones

# COMMAND ----------

worst_asset = tw1.impacto_65pb_total.idxmin()
best_asset = tw1.impacto_65pb_total.idxmax()
sig = int((tw1.t_total.abs() >= 2).sum())
cur, new = cmp2.iloc[0], cmp2.iloc[1]

summary = pd.DataFrame([
    {"id": "TW_01", "decision": "Mantener lectura salvo en activos con |t|≥2" if sig else "No cambiar la lectura: la sensibilidad a tasas no es estadísticamente distinguible de cero",
     "evidencia": f"{sig}/8 activos con |t|≥2; peor caso {names[worst_asset]} {tw1.impacto_65pb_total.min():+.1%}, mejor {names[best_asset]} {tw1.impacto_65pb_total.max():+.1%}; el choque es {SHOCK_BP / week_bp.abs().max():.0f}x lo observado"},
    {"id": "TW_02", "decision": "Dividir la cartera: 70% núcleo + 30% sleeve de baja volatilidad y alta liquidez",
     "evidencia": f"peor retorno a 6m {cur.ret_6m_peor:+.1%} (actual) vs {new.ret_6m_peor:+.1%} (dividida); drawdown {cur.max_drawdown:.1%} vs {new.max_drawdown:.1%} (tolerancia -10%)"},
    {"id": "TW_03", "decision": "El COP débil beneficia en pesos a un portafolio largo USD; revisar cobertura parcial si se quiere asegurar el 30% de liquidez",
     "evidencia": f"traslación +{SHOCK_LOW:.1%} a +{SHOCK_HIGH:.1%}; vol en COP {tw3.vol_cop.iloc[1]:.1%} vs {tw3.vol_usd.iloc[1]:.1%} en USD; cobertura de varianza mínima {tw3.ratio_cobertura_min_var.iloc[1]:.0%}"},
])
display(summary)

spark.createDataFrame(summary).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{T}.analysis_twist_summary")
