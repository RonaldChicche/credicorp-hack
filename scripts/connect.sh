#!/usr/bin/env bash
# Conecta tu máquina y GitHub con el workspace de Databricks.
# Uso: scripts/connect.sh https://dbc-xxxx.cloud.databricks.com [perfil]
set -euo pipefail

HOST="${1:?Falta el host del workspace, p. ej. https://dbc-xxxx.cloud.databricks.com}"
PROFILE="${2:-DEFAULT}"

# 1. Login OAuth (abre el navegador) y guarda el perfil en ~/.databrickscfg
databricks auth login --host "$HOST" --profile "$PROFILE"

# 2. Verifica identidad
databricks current-user me --profile "$PROFILE"

# 3. Valida el bundle contra el workspace
databricks bundle validate -t dev --profile "$PROFILE"

echo
echo "Listo. Siguiente: databricks bundle deploy -t dev --profile $PROFILE"
echo "Para CI, configura los secrets: gh secret set DATABRICKS_HOST / DATABRICKS_CLIENT_ID / DATABRICKS_CLIENT_SECRET"
