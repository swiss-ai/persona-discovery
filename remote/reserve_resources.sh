#!/bin/bash
# Reserve a node with the project container for interactive work. Submit from the repo root:
#   bash -c 'source remote/cluster.env && sbatch --account="$ACCOUNT" remote/reserve_resources.sh'

#SBATCH -J apertus-pulse
#SBATCH -t 06:00:00
#SBATCH --output=oreserve_resources.out
#SBATCH --error=esreserve_resources.err
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1

# Variables used by the entrypoint script
PROJECT_ROOT_AT=${PROJECT_ROOT_AT:-$SLURM_SUBMIT_DIR}
if [ ! -f "$PROJECT_ROOT_AT/remote/cluster.env" ]; then
  echo "[ERROR] $PROJECT_ROOT_AT/remote/cluster.env not found; submit from the repo root (template: remote/cluster.env.example)." >&2
  exit 1
fi
source "$PROJECT_ROOT_AT/remote/cluster.env"
export PROJECT_ROOT_AT
source $PROJECT_ROOT_AT/remote/setup.sh
export HF_TOKEN_AT=${HF_TOKEN_AT:-$HOME/.hf-token}
export SLURM_ONE_REMOTE_DEV=1
export PRE_COMMIT_HOME=$SCRATCH/pre-commit
export SSH_SERVER=1
export NO_SUDO_NEEDED=1

# Skip installing project
export SKIP_INSTALL_PROJECT=1

# VS Code server location
export VSCODE_SERVER_AT=$HOME/vscode-server
mkdir -p $VSCODE_SERVER_AT

echo $SCRATCH

srun \
  --container-image=$CONTAINER_IMAGE \
  --container-name=apertus-pulse \
  --environment=$CONTAINER_ENV_FILE \
  --container-mounts=\
"$(
  printf "%s," \
    "$PROJECT_ROOT_AT" \
    "$SCRATCH" \
    "${EXTRA_MOUNTS:-}" \
    "$WANDB_API_KEY_FILE_AT" \
    "$HF_TOKEN_AT" \
    "$HOME/.gitconfig" \
    "$HOME/.bashrc" \
    "$HOME/.oh-my-bash" \
    "$HOME/.ssh" \
    "$VSCODE_SERVER_AT" \
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
  sleep infinity

# To attach a shell once the job runs (job id from squeue):
#   srun --overlap --pty --jobid=<jobid> bash     # shell on the allocated node
#   enroot list -f                                # find the container's process id
#   enroot exec <container-pid> zsh               # shell inside the container

