#!/bin/bash
#
# parameter
# Explore different beam_size  K_prior/K_embed combinations
#
# Usage:
#   bash scripts/hyperparam_search.sh                    # new experiment
#   bash scripts/hyperparam_search.sh --resume <dir>     # resume from a checkpoint
#
# Examples:
#   nohup bash scripts/hyperparam_search.sh &
#   bash scripts/hyperparam_search.sh --resume results/hyperparam_search_20260112_171605
#

set -e

# to
cd "$(dirname "$0")/.."

# parameter
RESUME_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_DIR="$2"
            shift 2
            ;;
        *)
            echo "parameter: $1"
            exit 1
            ;;
    esac
completed

# Configure the output directory
if [ -n "$RESUME_DIR" ]; then
    OUTPUT_DIR="$RESUME_DIR"
    echo "Resume mode: $OUTPUT_DIR"
else
    OUTPUT_DIR="results/hyperparam_search_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$OUTPUT_DIR"
fi

LOG_FILE="${OUTPUT_DIR}/search.log"

# parameter
BEAM_SIZES=(1 3 5)
K_PRIOR_VALUES=(500 1000 2000)
K_EMBED_VALUES=(0 500 1000)

echo "" | tee -a "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"
echo "parameter: $(date)" | tee -a "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"
echo "Config:" | tee -a "$LOG_FILE"
echo "  - Sample: all" | tee -a "$LOG_FILE"
echo "  - Token length: unlimited" | tee -a "$LOG_FILE"
echo "  - BEAM_SIZES: ${BEAM_SIZES[*]}" | tee -a "$LOG_FILE"
echo "  - K_PRIOR_VALUES: ${K_PRIOR_VALUES[*]}" | tee -a "$LOG_FILE"
echo "  - K_EMBED_VALUES: ${K_EMBED_VALUES[*]}" | tee -a "$LOG_FILE"
echo "  - OUTPUT_DIR: $OUTPUT_DIR" | tee -a "$LOG_FILE"
if [ -n "$RESUME_DIR" ]; then
    echo "  - Mode: resume" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# Summary file
SUMMARY_FILE="${OUTPUT_DIR}/summary.csv"
if [ ! -f "$SUMMARY_FILE" ]; then
    echo "beam_size,K_prior,K_embed,best_accuracy,oracle_accuracy,elapsed_time" > "$SUMMARY_FILE"
fi

# Counters
TOTAL_EXPS=$(( ${#BEAM_SIZES[@]} * ${#K_PRIOR_VALUES[@]} * ${#K_EMBED_VALUES[@]} ))
CURRENT_EXP=0
SKIPPED=0

# Iterate over all combinations
for BEAM_SIZE in "${BEAM_SIZES[@]}"; do
    for K_PRIOR in "${K_PRIOR_VALUES[@]}"; do
        for K_EMBED in "${K_EMBED_VALUES[@]}"; do
            CURRENT_EXP=$((CURRENT_EXP + 1))
            
            # whether to completed
            EXP_OUTPUT="${OUTPUT_DIR}/beam${BEAM_SIZE}_kp${K_PRIOR}_ke${K_EMBED}"
            RESULT_FILE=$(ls -t "$EXP_OUTPUT"/results_*.json 2>/dev/null | head -1)
            
            if [ -n "$RESULT_FILE" ] && [ -f "$RESULT_FILE" ]; then
                #  summary.csv whether to 
                if grep -q "^${BEAM_SIZE},${K_PRIOR},${K_EMBED}," "$SUMMARY_FILE" 2>/dev/null; then
                    echo "[$CURRENT_EXP/$TOTAL_EXPS] skippedcompleted: beam=$BEAM_SIZE, K_prior=$K_PRIOR, K_embed=$K_EMBED" | tee -a "$LOG_FILE"
                    SKIPPED=$((SKIPPED + 1))
                    continue
                fi
            fi
            
            echo "" | tee -a "$LOG_FILE"
            echo "================================================" | tee -a "$LOG_FILE"
            echo "[$CURRENT_EXP/$TOTAL_EXPS] beam=$BEAM_SIZE, K_prior=$K_PRIOR, K_embed=$K_EMBED" | tee -a "$LOG_FILE"
            echo "Start time: $(date)" | tee -a "$LOG_FILE"
            echo "================================================" | tee -a "$LOG_FILE"
            
            # Run experiment
            mkdir -p "$EXP_OUTPUT"
            
            START_TIME=$(date +%s)
            
            python run.py \
                --beam_size "$BEAM_SIZE" \
                --K_prior "$K_PRIOR" \
                --K_embed "$K_EMBED" \
                --output_dir "$EXP_OUTPUT" \
                --log_dir "$EXP_OUTPUT" \
                --verbose \
                2>&1 | tee -a "$LOG_FILE"
            
            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            
            # Extract results
            RESULT_FILE=$(ls -t "$EXP_OUTPUT"/results_*.json 2>/dev/null | head -1)
            if [ -n "$RESULT_FILE" ]; then
                BEST_ACC=$(python -c "import json; d=json.load(open('$RESULT_FILE')); print(d['summary']['overall_accuracy'])" 2>/dev/null || echo "0")
                ORACLE_ACC=$(python -c "import json; d=json.load(open('$RESULT_FILE')); print(d['summary']['oracle_accuracy'])" 2>/dev/null || echo "0")
            else
                BEST_ACC="0"
                ORACLE_ACC="0"
            fi
            
            # Write summary
            echo "$BEAM_SIZE,$K_PRIOR,$K_EMBED,$BEST_ACC,$ORACLE_ACC,$ELAPSED" >> "$SUMMARY_FILE"
            
            echo "" | tee -a "$LOG_FILE"
            echo "completed: best_acc=$BEST_ACC, oracle_acc=$ORACLE_ACC, elapsed=${ELAPSED}s" | tee -a "$LOG_FILE"
        completed
    completed
completed

echo "" | tee -a "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"
echo "parametercompleted: $(date)" | tee -a "$LOG_FILE"
echo "skippedcompleted: $SKIPPED " | tee -a "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Summary:" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Print summary table
cat "$SUMMARY_FILE" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Detailed results saved in: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Summary CSV: $SUMMARY_FILE" | tee -a "$LOG_FILE"

# Find the best configuration
echo "" | tee -a "$LOG_FILE"
echo "Best configuration ( oracle_accuracy):" | tee -a "$LOG_FILE"
tail -n +2 "$SUMMARY_FILE" | sort -t',' -k5 -rn | head -3 | tee -a "$LOG_FILE"
