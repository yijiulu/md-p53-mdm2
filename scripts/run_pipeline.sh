#!/usr/bin/env bash
set -euo pipefail

# Portable wrapper for building a topology and running multi-replica OpenMM 8 production MD.
# Usage:
#   bash scripts/run_pipeline.sh <SYSTEM: 4XXB|2AC0_dimer> <MUTATIONS: WT|B:R326H|...> <VARIANT_LABEL> [REPLICAS: 1,2,3]
#
# Optional environment overrides:
#   PYTHON_BIN=python   PLATFORM=CUDA   TARGET_NS=200   CHUNK_NS=5   DT_FS=4.0

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
  echo "Usage: $0 <4XXB|2AC0_dimer> <MUTATIONS> <VARIANT_LABEL> [REPLICAS]" >&2
  exit 1
fi

SYSTEM="$1"
MUTATIONS="$2"
VARIANT="$3"
REPLICAS_CSV="${4:-1,2,3}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PLATFORM="${PLATFORM:-CUDA}"
TARGET_NS="${TARGET_NS:-200}"
CHUNK_NS="${CHUNK_NS:-5}"
DT_FS="${DT_FS:-4.0}"

RUN_DIR="${REPO_DIR}/runs/${SYSTEM}_${VARIANT}"
TIMINGS="${REPO_DIR}/runs/timings.csv"

mkdir -p "${REPO_DIR}/runs"
if [[ ! -f "$TIMINGS" ]]; then
  echo "system,variant,replica,start_utc,end_utc,elapsed_min" > "$TIMINGS"
fi

if [[ ! -f "${RUN_DIR}/build/solv_hmr.prmtop" ]]; then
  echo "[BUILD] ${SYSTEM} (${MUTATIONS}) -> ${RUN_DIR}"
  "$PYTHON_BIN" -u "${REPO_DIR}/scripts/01_build_system.py" \
    --system "$SYSTEM" \
    --mutations "$MUTATIONS" \
    --variant "$VARIANT" \
    --outdir "$RUN_DIR"
else
  echo "[SKIP] Reusing existing topology at ${RUN_DIR}/build/solv_hmr.prmtop"
fi

IFS=',' read -r -a REPLICAS <<< "$REPLICAS_CSV"
for rep in "${REPLICAS[@]}"; do
  start_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  start_s=$(date +%s)
  echo "[START] ${SYSTEM}_${VARIANT} rep${rep} at ${start_utc}"

  "$PYTHON_BIN" -u "${REPO_DIR}/scripts/02_run_md.py" \
    --run-dir "$RUN_DIR" \
    --replica "$rep" \
    --ns "$TARGET_NS" \
    --chunk-ns "$CHUNK_NS" \
    --dt-fs "$DT_FS" \
    --platform "$PLATFORM"

  end_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  end_s=$(date +%s)
  elapsed_min=$(awk -v s="$start_s" -v e="$end_s" 'BEGIN { printf "%.2f", (e-s)/60.0 }')
  echo "${SYSTEM},${VARIANT},${rep},${start_utc},${end_utc},${elapsed_min}" >> "$TIMINGS"
  echo "[DONE] ${SYSTEM}_${VARIANT} rep${rep} finished at ${end_utc} (${elapsed_min} min)"
done
