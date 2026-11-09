#!/bin/bash
# Submit the full per-trait pipeline as an afterok-chained set of Slurm jobs.
#
# Each trait gets its own independent chain:
#
#     2_prepare_probing_dataset  ->  3_train_probes  ->  5a_steer_trait  ->  6b_evaluate_steering
#
# With `afterok`, a stage only starts if the previous job of the trait exited 0, so a job that fails
# while running stops the rest of that trait's chain. A failed submission is not handled this way:
# the later stages of that trait are still submitted, depending on the last successfully submitted
# job (or on none). Traits are independent.
#
# Usage:
#   bash remote/submit_chain.sh <checkpoint> <traits> [stages]
#
#   <checkpoint>  e.g. apertus_mini_5B_base | apertus_mini_5B_instruct | apertus_v1.5_8B
#   <traits>      comma-separated or range, e.g. "1,2,3" or "1-30"
#   [stages]      comma-separated subset of: 2,3,5a,6b   (default: all four)
#
# Examples:
#   bash remote/submit_chain.sh apertus_mini_5B_base 1-3
#   bash remote/submit_chain.sh apertus_v1.5_8B 1-30 2,3
#   DRY_RUN=1 bash remote/submit_chain.sh apertus_v1.5_8B 1-30     # print, submit nothing
#
# Env:
#   DRY_RUN=1     show what would be submitted without submitting
#   FORCE_2A=1    submit stage 2 even if its activations already look complete

set -uo pipefail

# Personal / group settings (account, paths): remote/cluster.env, next to this script.
CLUSTER_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cluster.env"
if [ ! -f "$CLUSTER_ENV" ]; then
    echo "[ERROR] $CLUSTER_ENV not found. Copy remote/cluster.env.example to remote/cluster.env and fill in your values." >&2
    exit 1
fi
source "$CLUSTER_ENV"
PROJECT_ROOT="${PROJECT_ROOT_AT}"
cd "$PROJECT_ROOT" || exit 1

CHECKPOINT="${1:-}"
TRAIT_SPEC="${2:-}"
STAGES="${3:-2,3,5a,6b}"

if [ -z "$CHECKPOINT" ] || [ -z "$TRAIT_SPEC" ]; then
    echo "usage: bash remote/submit_chain.sh <checkpoint> <traits> [stages]" >&2
    echo "   e.g. bash remote/submit_chain.sh apertus_mini_5B_base 1-3" >&2
    exit 1
fi

# Expand "1-30" / "1,2,3" / "1-3,7,9-11" into a list.
expand_traits() {
    local spec="$1" out=""
    IFS=',' read -ra parts <<< "$spec"
    for p in "${parts[@]}"; do
        if [[ "$p" == *-* ]]; then
            out="$out $(seq "${p%-*}" "${p#*-}")"
        else
            out="$out $p"
        fi
    done
    echo $out
}

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

# Heuristic for "stage 2 already done": the file count of the reference setup (504 cal questions x
# 3 instruction pairs = 1512 prompts per sign -> 16 batches of 100, x 2 signs = 32 files per folder).
EXPECTED_BATCHES=32
acts_complete() {
    local trait="$1"
    local d="${SCRATCH}/apertus-pulse/data/cache/${CHECKPOINT}/trait_${trait}/cal/acts/layer_0/residual.post_mlp/mean_pooling"
    [ -d "$d" ] || return 1
    [ "$(find "$d" -name '*.pt' 2>/dev/null | wc -l)" -ge "$EXPECTED_BATCHES" ]
}

TRAITS=$(expand_traits "$TRAIT_SPEC")
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="${PROJECT_ROOT}/remote/logs/chain_${CHECKPOINT}_${STAMP}.txt"
mkdir -p "${PROJECT_ROOT}/remote/logs"

{
echo "chain submitted $(date)"
echo "checkpoint : $CHECKPOINT"
echo "traits     : $(echo $TRAITS | tr ' ' ',')"
echo "stages     : $STAGES"
echo "dry run    : ${DRY_RUN:-0}"
echo
printf "%-7s %-12s %-12s %-12s %-12s\n" "TRAIT" "2" "3" "5a" "6b"
} | tee "$LOG"

submit() {   # submit <dependency-or-empty> <args...>  -> echoes job id
    local dep="$1"; shift
    if [ -n "${DRY_RUN:-}" ]; then
        echo "DRY[$*${dep:+ after:$dep}]"
        return 0
    fi
    local id
    id=$(DEPEND_AFTEROK="$dep" PARSABLE=1 bash remote/submit_pipeline.sh "$@" 2>/dev/null | tail -1)
    if [[ "$id" =~ ^[0-9]+$ ]]; then echo "$id"; else echo "FAILED"; fi
}

n_sub=0
for t in $TRAITS; do
    j2a="-" ; j3="-" ; j6a="-" ; j6c="-"
    dep=""

    if has_stage 2; then
        if acts_complete "$t" && [ -z "${FORCE_2A:-}" ]; then
            j2a="skip(done)"        # activations already cached; chain starts at stage 3
        else
            j2a=$(submit "" -m src.2_prepare_probing_dataset trait_id="$t" model.checkpoint="$CHECKPOINT")
            [[ "$j2a" =~ ^[0-9]+$ ]] && { dep="$j2a"; n_sub=$((n_sub+1)); }
        fi
    fi

    if has_stage 3; then
        j3=$(submit "$dep" -m src.3_train_probes trait_id="$t" model.checkpoint="$CHECKPOINT")
        [[ "$j3" =~ ^[0-9]+$ ]] && { dep="$j3"; n_sub=$((n_sub+1)); }
    fi

    if has_stage 5a; then
        j6a=$(submit "$dep" -m src.5a_steer_trait trait_id="$t" model.checkpoint="$CHECKPOINT" probe_checkpoint="$CHECKPOINT")
        [[ "$j6a" =~ ^[0-9]+$ ]] && { dep="$j6a"; n_sub=$((n_sub+1)); }
    fi

    if has_stage 6b; then
        j6c=$(submit "$dep" -m src.6b_evaluate_steering trait_id="$t" checkpoint="$CHECKPOINT")
        [[ "$j6c" =~ ^[0-9]+$ ]] && n_sub=$((n_sub+1))
    fi

    printf "%-7s %-12s %-12s %-12s %-12s\n" "$t" "$j2a" "$j3" "$j6a" "$j6c" | tee -a "$LOG"
done

{
echo
echo "submitted $n_sub jobs; log: $LOG"
echo "watch:   squeue -u \$USER"
echo "cancel:  scancel -u \$USER -n prepare-probing   (or by job id from the table above)"
} | tee -a "$LOG"
