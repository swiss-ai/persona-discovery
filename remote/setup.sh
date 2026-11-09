# remote/setup.sh -- environment for the Slurm jobs; sourced by remote/submit_job.sh and
# remote/reserve_resources.sh after they have set PROJECT_ROOT_AT.
#
# Adapted from shared-submit-scripts/setup.sh of swiss-ai/posttraining (MIT License, see
# remote/LICENSE-posttraining). Personal and group values live in remote/cluster.env (gitignored;
# template: remote/cluster.env.example).

export SLURM_ONE_ENTRYPOINT_SCRIPT_PER_NODE=1
export HYDRA_FULL_ERROR=1
export PROJECT_NAME=apertus-pulse
export PACKAGE_NAME=apertus-pulse

if [ ! -f "${PROJECT_ROOT_AT}/remote/cluster.env" ]; then
  echo "[ERROR] ${PROJECT_ROOT_AT}/remote/cluster.env not found." >&2
  echo "        Copy remote/cluster.env.example to remote/cluster.env and fill in your values." >&2
  exit 1
fi
source "${PROJECT_ROOT_AT}/remote/cluster.env"
export CONTAINER_IMAGE
export CONTAINER_ENV_FILE="${PROJECT_ROOT_AT}/remote/edf.toml"

export HF_HOME=$SCRATCH/huggingface
export WANDB_API_KEY_FILE_AT=${WANDB_API_KEY_FILE_AT:-$HOME/.wandb-api-key}
export WANDB_ENTITY
export WANDB_PROJECT

export CUDA_BUFFER_PAGE_IN_THRESHOLD_MS=0.001
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
