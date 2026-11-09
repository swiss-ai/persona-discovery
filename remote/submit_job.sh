#!/bin/bash

# Job body for one pipeline stage; submitted by remote/submit_pipeline.sh, which passes the account
# and the log paths (from remote/cluster.env) as sbatch flags and PROJECT_ROOT_AT via --export.
#SBATCH -J apertus-job
#SBATCH -t 02:00:00
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 64G

set -euo pipefail

: "${PROJECT_ROOT_AT:?PROJECT_ROOT_AT is not set; submit jobs through remote/submit_pipeline.sh}"
if [ ! -f "$PROJECT_ROOT_AT/remote/cluster.env" ]; then
  echo "[ERROR] $PROJECT_ROOT_AT/remote/cluster.env not found (template: remote/cluster.env.example)." >&2
  exit 1
fi
source "$PROJECT_ROOT_AT/remote/cluster.env"

mkdir -p "$PROJECT_ROOT_AT/remote/logs"

# Core paths
export PROJECT_ROOT_AT
export HF_TOKEN_AT=${HF_TOKEN_AT:-$HOME/.hf-token}
export PRE_COMMIT_HOME=$SCRATCH/pre-commit
export NO_SUDO_NEEDED=1
export SKIP_INSTALL_PROJECT=1

source $PROJECT_ROOT_AT/remote/setup.sh

export HF_HOME=$SCRATCH/huggingface/job-tokens/${SLURM_JOB_ID}
mkdir -p "$HF_HOME"

srun \
  --container-image=$CONTAINER_IMAGE \
  --container-name=apertus-pulse-batch \
  --environment=$CONTAINER_ENV_FILE \
  --container-mounts="$(
    printf "%s," \
      "$PROJECT_ROOT_AT" \
      "$SCRATCH" \
      "${EXTRA_MOUNTS:-}" \
      "$WANDB_API_KEY_FILE_AT" \
      "$HF_TOKEN_AT" \
      "$HOME/.gitconfig" \
      "$HOME/.bashrc" \
      "$HOME/.local" \
      "$HOME/.ssh" \
      "$CONTAINER_IMAGE" \
      "$CONTAINER_ENV_FILE" \
    | tr ',' '\n' \
    | while read -r m; do
        [ -e "$m" ] && printf "%s," "$m"
      done \
    | sed 's/,$//'
  )" \
  --container-workdir=$PROJECT_ROOT_AT \
  --no-container-mount-home \
  --no-container-remap-root \
  --no-container-entrypoint \
  --container-writable \
  /opt/template-entrypoints/pre-entrypoint.sh \
  bash -lc '
    cd "$1"
    shift

    if [ -n "${EXTRA_PIP_PACKAGES:-}" ]; then
      echo "[INFO] Installing extra pip packages: ${EXTRA_PIP_PACKAGES}"
      python -m pip install --upgrade "${EXTRA_PIP_PACKAGES}"
    fi

    echo "[INFO] Current transformers version: $(python -c "import transformers; print(transformers.__version__)")"

    python "$@"
  ' _ "$PROJECT_ROOT_AT" "$@"