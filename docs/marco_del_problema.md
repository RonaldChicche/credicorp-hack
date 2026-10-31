# Marco del problema, métricas y score

Cómo se evalúa una cartera para un cliente: qué se quiere lograr, qué no se puede violar, qué se mide y cómo se llega al score.
El detalle de las tablas está en [`modelo_de_datos.md`](modelo_de_datos.md). Hoy solo CL_02 tiene configuración.

## 1. El problema

**Cliente:** CL_02, *Patrimonio Estable*. Moneda base COP, horizonte de 36 meses, tolerancia al riesgo Media-Baja,
necesidad de liquidez Media, prioridad "Preservación + crecimiento moderado".

**Twist del 17-ago-2026** (posterior al corte de datos del 14-ago):

| id | Novedad |
|---|---|
| TW_01 | El rendimiento del bono a 10 años de EE. UU. sube 65 pb en una semana |
| TW_02 | CL_02 necesitará disponer del 30% del portafolio en 6 meses |
| TW_03 | El COP se deprecia 8% frente al USD |

**Pregunta:** con las 8 acciones del dataset, ¿qué cartera le conviene a CL_02 y cumple lo que el cliente no puede aceptar?

**Alcance.** Solo se usa el dataset. Todas las acciones cotizan en USD; no hay efectivo, renta fija ni activos en COP.
Los retornos en COP se construyen combinando el retorno en USD con la serie `usd_cop` de `silver_macro`.

## 2. Objetivo

El twist le da al cliente dos horizontes, así que el objetivo se divide:

- **Tramo de 6 meses (30%):** que ese dinero esté disponible en COP sin pérdidas relevantes.
- **Tramo de 36 meses (70%):** crecimiento moderado dentro del límite de riesgo.

## 3. Límites del cliente

Son **filtros de aprobado o rechazado**, no puntos del score. Una cartera que incumple cualquiera queda **no apta** y no recibe score total.

| Límite | Valor | Origen | Cómo se mide |
|---|---|---|---|
| Drawdown máximo | 10% | Perfil del cliente | Peor caída desde un máximo, **en COP** |
| Peso en activos de alta volatilidad | 25% | Perfil del cliente | Suma de pesos de activos con volatilidad anual > 30% |
| Pérdida del tramo de 6 meses | 5% | Propuesta nuestra (TW_02) | Percentil 5 del retorno a 126 sesiones del tramo de liquidez, **en COP** |

Además, `NECESIDAD_6M = 30%` es un parámetro, no un límite: define qué fracción de la cartera forma el tramo de liquidez.

### Decisiones que fijan estos límites

- **El drawdown se mide en COP y no en USD.** La base del cliente es COP y el dinero se gasta en pesos; en USD se ignoraría el riesgo cambiario. El drawdown en USD queda como referencia.
- **El tramo de 6 meses se mide en COP.** Por la misma razón.
- **Alta volatilidad = más de 30% anual.** El perfil dice "máx. 25% en activos de alta volatilidad" sin definirla. Se prefirió un corte absoluto a uno relativo (tercio superior). Hoy son 4 de 8 activos: Pacifica Digital, Sierra Materials, Cordillera Energy y Norte Industrials.
- **La pérdida máxima del tramo de 6 meses (5%) es un criterio nuestro**, elegido para quedar por debajo del tope de la cartera completa. El dataset no lo define.

## 4. Métricas escogidas

Cinco indicadores, todos en la moneda base del cliente, agrupados por lo que le importa:

| Grupo | Peso | Indicador | Qué mide | Anclas (0 → 100 puntos) |
|---|---|---|---|---|
| Preservación | 35% | `dd_base` | Drawdown máximo | -10% → -5% |
| | | `vol_base` | Volatilidad anual | 20% → 10% |
| Liquidez a 6 meses | 30% | `p05_6m_base` | Percentil 5 del retorno a 126 sesiones del tramo de liquidez | -5% → -2.5% |
| Crecimiento | 20% | `ret_anual_base` | Retorno anual (media diaria × 252) | 0% → 8% |
| Riesgo externo | 15% | `razon_vol_base_usd` | Volatilidad en COP / volatilidad en USD | 1.5 → 1.0 |

**Por qué estos pesos.** La prioridad del perfil es preservar primero; el twist sube la liquidez al segundo lugar; el crecimiento es solo "moderado".
Son un criterio nuestro, sin respaldo estadístico.

**Por qué estas anclas.** Cuando hay un límite, 0 puntos es estar justo en el límite y 100 es tener la mitad del margen o más (`dd_base`, `p05_6m_base`).
Las anclas de `vol_base`, `ret_anual_base` y `razon_vol_base_usd` son arbitrarias: no hay límite del cual derivarlas.

### Métricas que se descartaron

- **Monto máximo vendible en 5 sesiones:** no se conoce el tamaño de la cartera, así que no hay contra qué compararlo.
- **Retorno / volatilidad:** repite lo que ya dicen el retorno y la volatilidad.
- **Impacto de +65 pb en tasas (TW_01):** ninguna sensibilidad de los 8 activos es estadísticamente distinta de cero (|t| < 2), y el choque es unas 17 veces la mayor variación semanal observada. Sirve para vigilar, no para decidir.

## 5. Cómo se decide el score

```
1. Filtro      cumple los 3 límites duros?  no -> apta = false, sin score total
2. Indicador   score = clip((valor - ancla_0) / (ancla_100 - ancla_0), 0, 1) x 100
3. Grupo       score_grupo = promedio de los scores de sus indicadores
4. Total       score_total = suma de (peso del grupo x score_grupo)
```

Entre las anclas el score sube de forma lineal; fuera del rango se recorta a 0 o a 100. Funciona igual si el mejor valor es el mayor (retorno) o el menor (volatilidad).

El score ordena carteras que ya pasaron el filtro. Un puntaje alto en un grupo no compensa un incumplimiento: por eso los límites no son parte del score.

### Carteras que se comparan

Hay dos carteras de referencia armadas con una heurística (abajo), y una **simulación** que busca las mejores (sección 7).

- `actual`: la cartera base de CL_02 (`gold_client_weights`), ponderada por inverso de la volatilidad sobre los activos elegibles. El 30% se obtendría vendiendo a prorrata.
- `dividida`: 70% de la cartera actual + 30% en un tramo de liquidez formado por los 3 activos con menor volatilidad, menor drawdown y mayor liquidez combinados. El ranking es por riesgo, no por retorno pasado.

## 6. Resultado actual

| Cartera | Drawdown en COP | Percentil 5 a 6 meses | Peso en alta volatilidad | Apta |
|---|---|---|---|---|
| `actual` | -11.1% | -2.0% | 0% | No |
| `dividida` | -10.96% | -1.0% | 0% | No |

Ninguna de estas dos carteras de referencia cumple el límite de -10% de drawdown en COP; la diferencia con el drawdown en USD la causa el tipo de cambio (la simulación de la sección 7 sí encuentra carteras que lo cumplen).
Por eso `gold_score_results.score_total` es nulo en ambas. Los scores por grupo sí se calculan:

| Cartera | Preservación | Liquidez 6m | Crecimiento | Riesgo externo |
|---|---|---|---|---|
| `actual` | 31.4 | 100 | 78.2 | 80.0 |
| `dividida` | 30.0 | 100 | 93.6 | 79.2 |

Como referencia, sin aplicar el filtro los totales serían ~69 y ~71; el sistema no los publica porque las carteras no son aptas.

## 7. Simulación de portafolios

`notebooks/score/02_simular_portafolios.py` responde qué carteras se le pueden recomendar. Cada cartera son dos vectores de pesos sobre las 8 acciones:
el **núcleo** (70%) y el **tramo de liquidez** (30%). El notebook:

1. Genera 100.000 carteras al azar (mezcla de concentradas y diversificadas) y las evalúa en COP con los indicadores y límites de la configuración.
2. Refina: perturba las 200 mejores durante 15 rondas y conserva las que mejoran.
3. Ordena: primero las que cumplen todos los límites, por `score_total`; si faltan, las de menor violación.

Antes de correr comprueba que su evaluación coincide con `gold_score_detail` hasta 1e-9, para que haya una sola definición de cada indicador.

**Resultado (semilla 42):** de 130.000 carteras evaluadas, **14.482 (11.1%) cumplen los tres límites**; el mejor drawdown hallado es -9.05% (el límite es -10%).
Es decir, el límite sí es alcanzable: lo que fallaba eran las dos carteras de referencia, no el límite. Las 50 mejores tienen score total entre 73.7 y 73.8.

**Cómo leerlo**
- Las 50 candidatas son casi intercambiables en score pero con pesos muy distintos: el score no identifica una única cartera óptima, solo una región de carteras buenas.
- Todo se calcula sobre el mismo año de datos con el que se eligen las carteras, así que el retorno histórico favorece a las acciones que mejor les fue (Aurora Health pesa mucho).
- No hay límite de concentración en la configuración: la mejor candidata pone el 69% del tramo de liquidez en una sola acción. Si el cliente no lo acepta, hay que agregar un límite de peso máximo por activo en `config_limits`.
- La búsqueda es aleatoria: no prueba que no exista una cartera mejor.

## 8. Cómo se aplica un twist nuevo

La configuración vive en `config/*.csv` y se carga a `config_limits`, `config_weights` y `config_indicators`. Un twist nuevo es agregar filas con otro `twist_id` y `vigente_desde`:
por cada clave gana la fila más reciente y las anteriores quedan como historial. El código no cambia.

Lo que sí requiere código es un **indicador nuevo** (hay que escribir su fórmula una vez) y un **cliente nuevo**, que necesita sus propios límites, pesos y anclas.

## 9. Limitaciones

- **Poca historia:** hay un año de datos diarios (ago-2025 a ago-2026) y 8 acciones. El score distingue poco entre carteras parecidas y las ventanas de 6 meses se solapan mucho.
- **Choques fuera de rango:** `usd_cop` se movió unos 6.5% en el año y el 10 años no superó los 30 pb en una semana. Los choques del twist son escenarios por extrapolación, no predicciones.
- **Intradía descartado:** los cierres de `silver_prices_intraday` difieren de los diarios ~14% en promedio, así que no se mezclan.
- **Sin activo sin riesgo:** el tramo de 6 meses se arma con acciones, no con efectivo o bonos.
- **Parámetros de criterio:** los pesos, las anclas arbitrarias y el límite de pérdida del tramo de 6 meses no salen de los datos; el proceso de selección del tramo de liquidez es dentro de la misma muestra.
- **Sin cobertura cambiaria:** el dataset no trae costos ni instrumentos de cobertura; solo se puede estimar cuánto reduciría la varianza.
