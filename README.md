# credicorp-hack

Databricks Asset Bundles (targets `dev` y `prod`) con despliegue por GitHub Actions.

## Conectores

| Conector | Cómo se configura |
|---|---|
| CLI local → workspace | `scripts/connect.sh <host> [perfil]` (OAuth, guarda `~/.databrickscfg`) |
| GitHub Actions → workspace | secrets `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET` (service principal OAuth M2M) |
| Workspace → GitHub (Repos/Git folders) | Settings > Linked accounts > GitHub, con un PAT |
| Workspace → S3 (Unity Catalog) | Storage credential (IAM role) + external location desde Catalog Explorer |

## Primeros pasos

1. Crea el workspace en AWS (console.databricks.com o AWS Marketplace) y copia su URL.
2. `aws login` y `scripts/connect.sh https://dbc-xxxx.cloud.databricks.com`
3. `databricks bundle deploy -t dev && databricks bundle run hello_job -t dev`
4. Para prod: crea un service principal, genera su secret OAuth y cárgalo con `gh secret set`.

## Estructura

- `databricks.yml`: bundle y targets
- `resources/`: jobs, pipelines, etc.
- `src/`: notebooks y código
- `.github/workflows/deploy.yml`: validate en PR, deploy a prod en `main`
- `notebooks/`: exploración + limpieza y análisis de portafolio
- `docs/modelo_de_datos.md`: modelo de datos de los CSV
- `notebooks/score/` + `config/`: score por cliente (límites duros, pesos y anclas configurables por twist)
