# Simulación de portafolios y recomendación

Cómo se buscaron las carteras para CL_02, cómo se califican, qué resultó y qué tan sólida es la recomendación.
El planteamiento, los límites y la fórmula del score están en [`marco_del_problema.md`](marco_del_problema.md); las tablas, en [`modelo_de_datos.md`](modelo_de_datos.md).

## 1. Qué se quería responder

Con las 8 acciones del dataset, ¿qué cartera se le puede recomendar a CL_02 (moneda base COP) de modo que cumpla sus límites duros?
No existe un optimizador cerrado para esto: el drawdown y el percentil 5 a 6 meses no son lineales en los pesos, así que se resolvió por **simulación**.

## 2. Cómo se hizo la simulación

Notebook: `notebooks/score/02_simular_portafolios.py` (tarea `simulacion` del job). Parámetros usados: 100.000 carteras, semilla 42, 15 rondas de refinamiento.

**Qué es una cartera.** Dos vectores de pesos sobre los 8 activos, sin posiciones cortas y sumando 1:
- el **núcleo**, que pesa 70%;
- el **tramo de liquidez**, que pesa 30% (`NECESIDAD_6M`, el dinero que el cliente necesitará en 6 meses).

El peso total de cada activo es `0.7 × núcleo + 0.3 × tramo`.

**Pasos**

1. **Muestreo.** Se generan 100.000 carteras al azar con distribuciones de Dirichlet, mezclando concentración alta, media y baja (alfa 0.3, 1 y 3), de modo que haya desde carteras con pocos activos hasta casi iguales. Núcleo y tramo se sortean por separado.
2. **Evaluación.** Cada cartera se evalúa en COP: el retorno diario en USD se combina con el cambio de `usd_cop`. Se calculan los 5 indicadores del score y la violación de los límites. Se evalúan miles a la vez con operaciones matriciales.
3. **Refinamiento.** Se toman las 200 mejores, cada una genera 10 hijas con ruido gaussiano (desviación 0.04 sobre los pesos, recortando a cero y renormalizando), y se conservan las 200 mejores entre padres e hijas. Se repite 15 veces.
4. **Orden.** Primero las que cumplen todos los límites, por `score_total` descendente; después las de menor violación. Se guardan las 50 mejores.

**Comprobación incorporada.** Antes de simular, el notebook evalúa las dos carteras de referencia con su propia función vectorizada y exige que los indicadores coincidan con `gold_score_detail` hasta 1e-9. Si no coinciden, aborta. Así hay una sola definición de cada indicador.

Todo lo configurable (límites, pesos, anclas, la fracción del tramo) se lee de `config_*`; el notebook no tiene números de negocio escritos.

## 3. Cómo se califica una cartera

**Paso 1, filtro.** La cartera debe cumplir los tres límites duros; si incumple alguno, es **no apta** y no recibe score total.

| Límite | Valor | Medido en |
|---|---|---|
| Drawdown máximo | 10% | COP |
| Peso en activos con volatilidad > 30% | 25% | Pesos totales |
| Pérdida del tramo de 6 meses (percentil 5 a 126 sesiones) | 5% | COP, solo del tramo de liquidez |

**Paso 2, score.** Cada indicador se lleva a 0–100 entre dos anclas, el grupo es el promedio de sus indicadores y el total pondera los grupos:

```
score_indicador = clip((valor - ancla_0) / (ancla_100 - ancla_0), 0, 1) × 100
score_total     = Σ peso_grupo × promedio(score_indicador del grupo)
```

| Grupo | Peso | Indicador | 0 puntos | 100 puntos |
|---|---|---|---|---|
| Preservación | 35% | Drawdown máximo en COP | -10% | -5% |
| | | Volatilidad anual en COP | 20% | 10% |
| Liquidez a 6 meses | 30% | Percentil 5 a 6 meses del tramo, en COP | -5% | -2.5% |
| Crecimiento | 20% | Retorno anual en COP | 0% | 8% |
| Riesgo externo | 15% | Volatilidad COP / volatilidad USD | 1.5 | 1.0 |

El filtro y el score están separados a propósito: un buen puntaje en un grupo no compensa un incumplimiento. Los pesos de grupo y las anclas de volatilidad, crecimiento y riesgo externo son criterio nuestro, no salen de los datos.

## 4. Resultados de la simulación

Semilla 42, tabla `gold_sim_summary`:

| | |
|---|---|
| Carteras evaluadas | 130.000 (100.000 muestreadas + 30.000 del refinamiento) |
| Carteras que cumplen los tres límites | 14.482 (11.1%) |
| Mejor drawdown hallado | -9.05% (el límite es -10%) |
| Score total de las 50 mejores | entre 73.47 y 73.84 |

**Lectura.**
- El límite sí es alcanzable: las dos carteras de referencia armadas con heurística (`actual` y `dividida`, ambas con drawdown de -11.1% y -10.96%) fallaban, pero hay carteras que lo cumplen.
- La región factible es angosta: solo 1 de cada 9 cumple, y el mejor drawdown posible (-9.05%) queda a menos de un punto del límite.
- Las 50 mejores son casi intercambiables en score pero con pesos muy distintos. El score señala una región de buenas carteras, no una única óptima.

## 5. Cartera recomendada

Se escogió la candidata **#3** (`gold_sim_candidates.sim_id = 3`). No es la de mayor score: el score entre las mejores casi empata, así que el desempate fue por **menor concentración**. La #1 pone el 69% del tramo de liquidez en una sola acción; en la #3 ningún activo pasa del 27.8% de la cartera ni del 31.2% del tramo.

| Activo | Sector / país | Peso total | Núcleo | Tramo de liquidez |
|---|---|---|---|---|
| Aurora Health | Salud, EE. UU. | 27.8% | 28.8% | 25.4% |
| Atlas Utilities | Servicios públicos, EE. UU. | 25.9% | 29.7% | 17.1% |
| Lima Financial | Financiero, Perú | 19.3% | 14.3% | 31.2% |
| Andes Consumer | Consumo básico, Colombia | 13.9% | 10.1% | 22.8% |
| Norte Industrials | Industriales, México | 8.3% | 11.7% | 0.5% |
| Pacifica Digital | Tecnología, Perú | 4.7% | 5.4% | 3.1% |

Sierra Materials y Cordillera Energy quedan en 0%.

| Límite | Valor | Límite | Margen |
|---|---|---|---|
| Drawdown máximo en COP | -9.34% | -10% | 0.66 puntos |
| Percentil 5 del tramo a 6 meses | +0.36% | -5% | Holgado |
| Peso en alta volatilidad | 13.0% | 25% | Holgado |

Retorno anual en COP de 8.5% y volatilidad de 14.8%. Score total **73.8**: preservación 32.6, liquidez 100, crecimiento 100, riesgo externo 82.7.

**Cómo se asignó cada peso.** No hay una fórmula que justifique cada peso por separado. Salen de la búsqueda: "esta combinación puntúa bien en conjunto y cumple los límites". No hay un score por activo.

## 6. Justificaciones: pruebas sobre la cartera recomendada

Estas pruebas se corrieron como un script aparte, sobre las tablas `silver_*` y `gold_*`, con los mismos límites, indicadores y moneda (COP). **No están en el repo**: no hay un notebook que las reproduzca.

### 6.1 Frente a alternativas simples

| Cartera | Drawdown en COP | Percentil 5 a 6 meses | Retorno anual | ¿Apta? |
|---|---|---|---|---|
| **Recomendada (#3)** | **-9.3%** | +0.4% | 8.5% | **Sí** |
| Base `actual` | -11.1% | -2.0% | 6.3% | No |
| Base `dividida` | -11.0% | -1.0% | 7.5% | No |
| Inverso a la volatilidad, 8 activos | -13.9% | -8.3% | -1.2% | No |
| Igual ponderada, 8 activos | -17.2% | -10.2% | -2.1% | No |
| Solo Aurora Health | -14.6% | +0.7% | 31.3% | No |

### 6.2 Aporte de cada activo
Se quita un activo y su peso se reparte entre los demás:

| Quito | Efecto en drawdown | Efecto en retorno | Efecto en volatilidad | Sigue apta |
|---|---|---|---|---|
| Aurora Health | -2.5 pts | -8.9 pts | +0.2 pts | No |
| Lima Financial | -2.2 pts | -3.6 pts | -0.4 pts | No |
| Norte Industrials | -1.3 pts | +2.1 pts | 0.0 pts | No |
| Pacifica Digital | -0.9 pts | +0.5 pts | 0.0 pts | No |
| Atlas Utilities | -0.8 pts | +6.6 pts | +2.9 pts | No |
| Andes Consumer | 0.0 pts | +2.5 pts | +1.2 pts | **Sí** |

Cinco de los seis activos son necesarios para cumplir el límite: la diversificación es lo que lo hace posible. Atlas Utilities cuesta unos 6.6 puntos de retorno y se mantiene porque baja el riesgo. Andes Consumer es el único que no aporta al drawdown, así que es el peso menos justificado.

### 6.3 Estabilidad
- **Por mitades del año:** el drawdown es casi idéntico (-9.34% y -9.33%), pero el retorno anualizado baja de +11.9% a +5.1%.
- **Días de estrés** (10 días con `risk_regime = Stress`): -6.1% acumulado, frente a -7.95% de la cartera igual ponderada.

### 6.4 Robustez al orden de los días
Se remuestrearon los mismos 270 días en bloques de 10, 5.000 veces, y se midió el drawdown de cada trayectoria:

| Cartera | Prob. de cumplir -10% | Drawdown mediano | Peor 5% |
|---|---|---|---|
| **Recomendada** | **31%** | -12.0% | -22.7% |
| Inverso a la volatilidad | 15% | -15.4% | -28.4% |
| Igual ponderada | 8% | -17.9% | -32.6% |
| Solo Aurora | 4% | -17.7% | -32.7% |

La recomendada es mucho mejor que las alternativas, pero cumple el límite solo en 3 de cada 10 reordenamientos: el "apta" de la muestra depende en parte del orden en que ocurrieron los días.

### 6.5 Estrés cambiario
Hoy el drawdown de la cartera es -4.5%. Si el USD/COP baja 7.4% (la reversión del choque del twist), el drawdown máximo pasa a **-11.55%** e incumple. Con una baja de 3% se queda en -9.34%.

### 6.6 Con cobertura cambiaria supuesta
Se repitió con una cobertura que anula una parte del movimiento de `usd_cop`, **sin costo**, porque el dataset no trae costos ni instrumentos:

| Cobertura | Drawdown histórico | Tras reversión del COP | Prob. de cumplir -10% |
|---|---|---|---|
| 0% | -9.3% | -11.6% (incumple) | 31% |
| 50% | -8.8% | -8.8% (cumple) | 36% |
| 100% | -8.3% | -8.3% (cumple) | 38% |

La cobertura resuelve el escenario de reversión del COP, pero casi no mejora la robustez general: la fragilidad viene sobre todo del riesgo de las acciones y no del tipo de cambio.

## 7. Conclusión

- **Se defiende como** la mejor cartera factible encontrada bajo los límites de CL_02 con estas 8 acciones: es la única apta frente a las alternativas probadas, necesita casi todos sus activos y está cerca del mejor drawdown posible.
- **No se defiende como** "la mejor decisión de inversión" ni como segura: el límite de -10% queda casi al borde de lo alcanzable, la robustez al orden de los días es de solo 31% y falla ante una reversión del COP.
- **Lo que la haría más sólida:** presentarla con una cobertura cambiaria parcial (50% elimina el incumplimiento por reversión) y decirle al cliente que cumplir -10% solo con acciones deja un margen estrecho. Otras salidas son un límite algo más alto o un activo sin riesgo, que el dataset no tiene.

## 8. Limitaciones

- **Un año de datos** y 8 acciones, y la cartera se eligió con la misma muestra con la que se evalúa: está ajustada a ese año, que favoreció a Aurora Health.
- **La búsqueda es aleatoria.** Encuentra buenas carteras pero no prueba que sean óptimas ni que no exista una mejor. Otra semilla daría otras 50 candidatas.
- **No hay límite de concentración** en la configuración. La recomendación evita la concentración por un criterio de desempate, no por una regla.
- **Sin activo sin riesgo:** el tramo de 6 meses se arma con acciones, cuyo precio puede caer.
- **Los choques del twist son escenarios por extrapolación:** `usd_cop` se movió unos 6.5% en el año y el 10 años no superó los 30 pb en una semana.
- **La cobertura se supuso sin costo ni instrumento concreto.**
- **Pesos de grupo, anclas arbitrarias y el límite de pérdida del tramo (5%)** son criterio nuestro.
- **Esto no es asesoría de inversión;** es una recomendación dentro de este marco y con estos datos.

## 9. Cómo reproducirlo

```
databricks bundle deploy -t dev
databricks bundle run pipeline_job -t dev --only simulacion
```

Widgets del notebook: `cliente` (`CL_02`), `n_sim` (`100000`), `rondas_refinamiento` (`15`), `top_n` (`50`), `seed` (`42`). Con otra semilla o más simulaciones las candidatas cambian; la corrida documentada aquí es la de semilla 42.
