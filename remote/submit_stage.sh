#!/bin/bash
# Submit one pipeline stage for a range of traits, dependency-aware.
#
# Slurm forgets finished jobs after a site-dependent time (MinJobAge; 300 s on the cluster used for
# the thesis), after which `--dependency=afterok:<jobid>` on such a job is rejected. This script
# therefore records the jobs it submits in remote/logs/job_registry.tsv, looks up the recorded
# prerequisite job of each trait with sacct, and decides:
#
#   PENDING / RUNNING   -> submit with --dependency=afterok:<id>
#   COMPLETED           -> submit without a dependency (already satisfied)
#   FAILED/TIMEOUT/...  -> skip and report it (do not run on missing input)
#   unknown to sacct    -> skip and report it (success cannot be verified)
#
# Usage:
#   bash remote/submit_stage.sh <checkpoint> <stage> <prev_stage> <traits> [walltime]
#     stage / prev_stage in: 2, 3, 5a, 6b   (prev_stage "-" means no prerequisite)
#
#   bash remote/submit_stage.sh apertus_mini_5B_instruct 5a 3 2-30
#   bash remote/submit_stage.sh apertus_v1.5_8B 6b 5a 2-30 00:15:00

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
REG="${PROJECT_ROOT}/remote/logs/job_registry.tsv"
mkdir -p "$(dirname "$REG")" && touch "$REG"   # fresh checkout: start with an empty registry

CK="${1:?checkpoint}"; STAGE="${2:?stage}"; PREV="${3:?prev stage or -}"; SPEC="${4:?traits}"; WALL="${5:-}"

case "$STAGE" in
  2)  MOD="src.2_prepare_probing_dataset";  ARGS=(model.checkpoint="$CK") ;;
  3)  MOD="src.3_train_probes";             ARGS=(model.checkpoint="$CK") ;;
  5a) MOD="src.5a_steer_trait";             ARGS=(model.checkpoint="$CK" probe_checkpoint="$CK") ;;
  6b) MOD="src.6b_evaluate_steering";       ARGS=(checkpoint="$CK") ;;
  *)  echo "unknown stage: $STAGE" >&2; exit 1 ;;
esac

expand() { local o=""; IFS=',' read -ra p <<< "$1"
  for x in "${p[@]}"; do if [[ "$x" == *-* ]]; then o="$o $(seq "${x%-*}" "${x#*-}")"; else o="$o $x"; fi; done; echo $o; }

n=0; skipped=0; nodep=0
declare -a PROBLEMS=()

for t in $(expand "$SPEC"); do
    # already submitted for this (ck, trait, stage)?
    if awk -v c="$CK" -v tt="$t" -v s="$STAGE" '$1==c && $2==tt && $3==s{f=1} END{exit !f}' "$REG"; then
        continue
    fi

    dep=""
    if [ "$PREV" != "-" ]; then
        pid=$(awk -v c="$CK" -v tt="$t" -v s="$PREV" '$1==c && $2==tt && $3==s{print $4}' "$REG")
        if [ -z "$pid" ]; then PROBLEMS+=("trait $t: no $PREV job recorded"); skipped=$((skipped+1)); continue; fi
        st=$(sacct -j "$pid" -X --format=State -P -n 2>/dev/null | head -1 | sed 's/ by .*//')
        case "$st" in
            PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|REQUEUED) dep="$pid" ;;
            COMPLETED)                                                 dep=""; nodep=$((nodep+1)) ;;
            "")        PROBLEMS+=("trait $t: $PREV job $pid unknown to sacct"); skipped=$((skipped+1)); continue ;;
            *)         PROBLEMS+=("trait $t: $PREV job $pid is $st");           skipped=$((skipped+1)); continue ;;
        esac
    fi

    id=$(TIME_LIMIT_OVERRIDE="$WALL" DEPEND_AFTEROK="$dep" PARSABLE=1 \
         bash remote/submit_pipeline.sh -m "$MOD" trait_id="$t" "${ARGS[@]}" 2>/dev/null | tail -1)
    if [[ "$id" =~ ^[0-9]+$ ]]; then
        printf "%s\t%s\t%s\t%s\n" "$CK" "$t" "$STAGE" "$id" >> "$REG"; n=$((n+1))
    else
        PROBLEMS+=("trait $t: sbatch failed"); skipped=$((skipped+1))
    fi
done

echo "$CK stage $STAGE: submitted $n (of which $nodep with prerequisite already COMPLETED), skipped $skipped"
for p in "${PROBLEMS[@]:-}"; do if [ -n "$p" ]; then echo "    ! $p"; fi; done
