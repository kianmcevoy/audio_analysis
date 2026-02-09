#!/usr/bin/env bash
set -euo pipefail
bash setup.sh
source .venv/bin/activate

BUNDLE_ROOT="${1:?usage: analyse_bundle.sh <analysis_runs/run_folder>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

python -m analyse.cli bundle --input "${BUNDLE_ROOT}"
