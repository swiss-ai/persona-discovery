#!/bin/bash
# Submit one pipeline stage as a Slurm job (body: remote/submit_job.sh), with the job resources of that stage.
#
# Usage:
#   bash remote/submit_pipeline.sh [--debug] -m src.<stage> [hydra overrides | script options ...]
#
#   bash remote/submit_pipeline.sh -m src.0_dump_questions
#   bash remote/submit_pipeline.sh -m src.3_train_probes trait_id=7 model.checkpoint=apertus_v1.5_8B
#
# Env: DEPEND_AFTEROK=<jobid>[:<jobid>...] (start after those jobs succeed), PARSABLE=1 (print only the job id),
#      TIME_LIMIT_OVERRIDE=HH:MM:SS. --debug submits a short, small test job.

set -euo pipefail

# Personal / group settings (account, paths): remote/cluster.env, next to this script.
CLUSTER_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cluster.env"
if [ ! -f "$CLUSTER_ENV" ]; then
    echo "[ERROR] $CLUSTER_ENV not found. Copy remote/cluster.env.example to remote/cluster.env and fill in your values." >&2
    exit 1
fi
source "$CLUSTER_ENV"
PROJECT_ROOT="${PROJECT_ROOT_AT}"
SUBMIT_JOB_SCRIPT="${PROJECT_ROOT}/remote/submit_job.sh"

if [ ! -f "${SUBMIT_JOB_SCRIPT}" ]; then
    echo "[ERROR] submit_job.sh not found at: ${SUBMIT_JOB_SCRIPT}"
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "[ERROR] No Python target provided."
    exit 1
fi

DEBUG_MODE="false"
FORWARDED_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --debug)
            DEBUG_MODE="true"
            ;;
        *)
            FORWARDED_ARGS+=("$arg")
            ;;
    esac
done

if [ ${#FORWARDED_ARGS[@]} -eq 0 ]; then
    echo "[ERROR] No Python target provided after parsing flags."
    exit 1
fi

TARGET_KIND=""
TARGET_KEY=""

if [ "${FORWARDED_ARGS[0]}" = "-m" ]; then
    if [ ${#FORWARDED_ARGS[@]} -lt 2 ]; then
        echo "[ERROR] '-m' was provided but no module name followed."
        exit 1
    fi

    TARGET_KIND="module"
    TARGET_KEY="${FORWARDED_ARGS[1]}"   # e.g. src.2_prepare_probing_dataset
else
    TARGET_KIND="file"
    TARGET_KEY="$(basename "${FORWARDED_ARGS[0]}" .py)"   # e.g. 0_dump_questions
fi

JOB_NAME=""
TIME_LIMIT=""
MEMORY=""
CPUS_PER_TASK=""
EXTRA_PIP_PACKAGES=""
TRANSFORMERS_REQUIREMENT='transformers @ git+https://github.com/swiss-ai/transformers.git@3797303dda74844e3d1f8977ff5518bb91f818b4'



# Job resources per stage. The time limits are set from approximate run times on the reference setup.
case "${TARGET_KEY}" in
    0_dump_questions|src.0_dump_questions)
        JOB_NAME="dump-questions"
        TIME_LIMIT="00:05:00"
        MEMORY="8G"
        CPUS_PER_TASK="4"
        EXTRA_PIP_PACKAGES="fire"
        ;;
    1a_prepare_traits|src.1a_prepare_traits)
        JOB_NAME="prepare-traits"
        TIME_LIMIT="00:10:00"
        MEMORY="8G"
        CPUS_PER_TASK="4"
        ;;
    1b_prepare_instructions|src.1b_prepare_instructions)
        JOB_NAME="prepare-instr"
        TIME_LIMIT="00:10:00"
        MEMORY="8G"
        CPUS_PER_TASK="4"
        ;;
    2_prepare_probing_dataset|src.2_prepare_probing_dataset)
        JOB_NAME="prepare-probing"
        TIME_LIMIT="00:25:00"
        MEMORY="64G"
        CPUS_PER_TASK="16"
        EXTRA_PIP_PACKAGES="$TRANSFORMERS_REQUIREMENT"
        ;;
    3_train_probes|src.3_train_probes)
        JOB_NAME="train-probes"
        TIME_LIMIT="00:15:00"
        MEMORY="6G"
        CPUS_PER_TASK="4"
        EXTRA_PIP_PACKAGES="$TRANSFORMERS_REQUIREMENT"
        ;;
    5a_steer_trait|src.5a_steer_trait)
        JOB_NAME="steer-trait"
        TIME_LIMIT="00:20:00"
        MEMORY="96G"
        CPUS_PER_TASK="16"
        EXTRA_PIP_PACKAGES="$TRANSFORMERS_REQUIREMENT"
        ;;
    6a_score_coherence|src.6a_score_coherence)
        JOB_NAME="score-coherence"
        TIME_LIMIT="00:10:00"   # CPU-only, no model
        MEMORY="8G"
        CPUS_PER_TASK="4"
        ;;
    5b_steer_angular|src.5b_steer_angular)
        JOB_NAME="steer-angular"
        TIME_LIMIT="00:25:00"
        MEMORY="96G"
        CPUS_PER_TASK="16"
        EXTRA_PIP_PACKAGES="$TRANSFORMERS_REQUIREMENT"
        ;;
    6b_evaluate_steering|src.6b_evaluate_steering)
        JOB_NAME="eval-steering"
        TIME_LIMIT="00:40:00"   # long enough for list-valued eval_trait_id
        MEMORY="24G"
        CPUS_PER_TASK="8"
        ;;
    *)
        JOB_NAME="${TARGET_KEY}"
        JOB_NAME="${JOB_NAME//./-}"
        JOB_NAME="${JOB_NAME//_/-}"
        TIME_LIMIT="01:00:00"
        CPUS_PER_TASK="4"
        MEMORY="32G"
        ;;
esac

if [ "${DEBUG_MODE}" = "true" ]; then
    TIME_LIMIT="00:03:00"
    MEMORY="2G"
    CPUS_PER_TASK="1"
    JOB_NAME="dbg-${JOB_NAME}"
fi

# Per-submission walltime override, e.g. TIME_LIMIT_OVERRIDE=00:30:00 for slower checkpoints.
[ -n "${TIME_LIMIT_OVERRIDE:-}" ] && TIME_LIMIT="${TIME_LIMIT_OVERRIDE}"

TRAIT_PREFIX=""

for arg in "${FORWARDED_ARGS[@]}"; do
    case "$arg" in
        trait_id=*)
            TRAIT_VALUE="${arg#trait_id=}"
            TRAIT_VALUE="${TRAIT_VALUE//,/\-}"
            TRAIT_VALUE="${TRAIT_VALUE// /}"
            TRAIT_PREFIX="t${TRAIT_VALUE}-"
            break
            ;;
    esac
done

JOB_NAME="${TRAIT_PREFIX}${JOB_NAME}"

# To stderr when PARSABLE, so stdout carries only the job id for chaining.
INFO_LINE="[INFO] DEBUG_MODE=${DEBUG_MODE} | JOB_NAME=${JOB_NAME} | TIME_LIMIT=${TIME_LIMIT} | CPUS_PER_TASK=${CPUS_PER_TASK} | MEMORY=${MEMORY}"
if [ -n "${PARSABLE:-}" ]; then echo "$INFO_LINE" >&2; else echo "$INFO_LINE"; fi

# Optional DEPEND_AFTEROK / PARSABLE (see the header).
EXTRA_SBATCH=()
[ -n "${DEPEND_AFTEROK:-}" ] && EXTRA_SBATCH+=(--dependency="afterok:${DEPEND_AFTEROK}")
[ -n "${PARSABLE:-}" ] && EXTRA_SBATCH+=(--parsable)

# Account and log paths come from cluster.env (#SBATCH lines cannot read variables);
# PROJECT_ROOT_AT tells submit_job.sh where the repo and its cluster.env are.
# Slurm does not create the log directory, so it must exist before submission.
mkdir -p "${PROJECT_ROOT}/remote/logs"
sbatch \
    --job-name="${JOB_NAME}" \
    --time="${TIME_LIMIT}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEMORY}" \
    --account="${ACCOUNT}" \
    --output="${PROJECT_ROOT}/remote/logs/%x-%j.out" \
    --error="${PROJECT_ROOT}/remote/logs/%x-%j.err" \
    --export=ALL,EXTRA_PIP_PACKAGES="${EXTRA_PIP_PACKAGES}",PROJECT_ROOT_AT="${PROJECT_ROOT}" \
    "${EXTRA_SBATCH[@]}" \
    "${SUBMIT_JOB_SCRIPT}" \
    "${FORWARDED_ARGS[@]}"