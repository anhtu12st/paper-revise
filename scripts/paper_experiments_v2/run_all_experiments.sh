#!/bin/bash
################################################################################
# Run All Hybrid Experiments
################################################################################
#
# This script runs all 12 experiments (6 Standard SQuAD v2 + 6 Long SQuAD v2),
# evaluates them, and generates publication-quality figures.
#
# Usage:
#   bash scripts/paper_experiments_v2/run_all_experiments.sh
#
# Optional flags:
#   --skip-training      Skip training, only run evaluation and analysis
#   --skip-evaluation    Skip evaluation, only run training
#   --skip-analysis      Skip analysis, only run training and evaluation
#   --experiments "01 02 03"  Run only specific experiments (space-separated)
#
# Examples:
#   # Run everything
#   bash scripts/paper_experiments_v2/run_all_experiments.sh
#
#   # Run only experiments 01, 02, and evaluation
#   bash scripts/paper_experiments_v2/run_all_experiments.sh --experiments "01 02"
#
#   # Skip training, only evaluate and analyze existing results
#   bash scripts/paper_experiments_v2/run_all_experiments.sh --skip-training
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse command-line arguments
SKIP_TRAINING=false
SKIP_EVALUATION=false
SKIP_ANALYSIS=false
SPECIFIC_EXPERIMENTS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-evaluation)
            SKIP_EVALUATION=true
            shift
            ;;
        --skip-analysis)
            SKIP_ANALYSIS=true
            shift
            ;;
        --experiments)
            SPECIFIC_EXPERIMENTS="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
SQUAD_DIR="$SCRIPT_DIR/squad"
LONG_SQUAD_DIR="$SCRIPT_DIR/long_squad"
LOG_DIR="$PROJECT_ROOT/outputs/paper_v2_logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Experiment lists
SQUAD_EXPERIMENTS=(
    "01_baseline_squad_no_memory"
    "02_main_squad_8tokens"
    "03_ablation_no_gating"
    "04_ablation_4tokens"
    "05_ablation_16tokens"
    "06_ablation_32tokens"
)

LONG_SQUAD_EXPERIMENTS=(
    "07_baseline_no_memory"
    "08_main_8tokens"
    "09_ablation_no_progressive"
    "10_ablation_no_gating"
    "11_segments_4seg"
    "12_segments_6seg"
)

# Combine all experiments
ALL_EXPERIMENTS=("${SQUAD_EXPERIMENTS[@]}" "${LONG_SQUAD_EXPERIMENTS[@]}")

# Filter experiments if specific ones requested
if [ -n "$SPECIFIC_EXPERIMENTS" ]; then
    echo -e "${YELLOW}Running only experiments: $SPECIFIC_EXPERIMENTS${NC}"
    FILTERED_EXPERIMENTS=()
    for exp_id in $SPECIFIC_EXPERIMENTS; do
        # Find matching experiment
        for exp in "${ALL_EXPERIMENTS[@]}"; do
            if [[ $exp == ${exp_id}_* ]]; then
                FILTERED_EXPERIMENTS+=("$exp")
            fi
        done
    done
    ALL_EXPERIMENTS=("${FILTERED_EXPERIMENTS[@]}")
fi

# Function to print section headers
print_header() {
    echo ""
    echo -e "${PURPLE}================================================================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}================================================================================${NC}"
    echo ""
}

# Function to print experiment info
print_experiment() {
    local exp_num=$1
    local exp_name=$2
    local dataset=$3
    echo ""
    echo -e "${CYAN}--------------------------------------------------------------------------------${NC}"
    echo -e "${CYAN}Experiment $exp_num: $exp_name${NC}"
    echo -e "${CYAN}Dataset: $dataset${NC}"
    echo -e "${CYAN}--------------------------------------------------------------------------------${NC}"
}

# Function to run an experiment
run_experiment() {
    local exp_file=$1
    local exp_dir=$2
    local exp_name=$(basename "$exp_file" .py)
    local exp_num=$(echo "$exp_name" | grep -o '^[0-9]\+')
    local log_file="$LOG_DIR/${exp_name}.log"

    # Determine dataset type
    local dataset="Standard SQuAD v2"
    if [ "$exp_dir" = "$LONG_SQUAD_DIR" ]; then
        dataset="Long SQuAD v2"
    fi

    print_experiment "$exp_num" "$exp_name" "$dataset"

    echo -e "${BLUE}▶ Starting training...${NC}"
    echo -e "${BLUE}▶ Log file: $log_file${NC}"

    # Run experiment
    if uv run python "$exp_file" 2>&1 | tee "$log_file"; then
        echo -e "${GREEN}✅ Completed: $exp_name${NC}"
        return 0
    else
        echo -e "${RED}❌ Failed: $exp_name${NC}"
        echo -e "${RED}   Check log: $log_file${NC}"
        return 1
    fi
}

# Start timing
START_TIME=$(date +%s)

print_header "🚀 HYBRID EXPERIMENTS - TRAINING, EVALUATION & ANALYSIS"

echo -e "${BLUE}Project root: $PROJECT_ROOT${NC}"
echo -e "${BLUE}Script directory: $SCRIPT_DIR${NC}"
echo -e "${BLUE}Log directory: $LOG_DIR${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "${BLUE}  Skip training: $SKIP_TRAINING${NC}"
echo -e "${BLUE}  Skip evaluation: $SKIP_EVALUATION${NC}"
echo -e "${BLUE}  Skip analysis: $SKIP_ANALYSIS${NC}"
echo -e "${BLUE}  Total experiments: ${#ALL_EXPERIMENTS[@]}${NC}"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# ============================================================================
# TRAINING PHASE
# ============================================================================

if [ "$SKIP_TRAINING" = false ]; then
    print_header "📚 PHASE 1: TRAINING"

    COMPLETED_COUNT=0
    FAILED_COUNT=0
    FAILED_EXPERIMENTS=()

    # Standard SQuAD v2 experiments
    print_header "📊 Standard SQuAD v2 Experiments (1-2 segments)"

    for exp_file in "${SQUAD_EXPERIMENTS[@]}"; do
        # Skip if not in filtered list
        if [ -n "$SPECIFIC_EXPERIMENTS" ]; then
            exp_num=$(echo "$exp_file" | grep -o '^[0-9]\+')
            if ! echo "$SPECIFIC_EXPERIMENTS" | grep -q "$exp_num"; then
                continue
            fi
        fi

        if run_experiment "$SQUAD_DIR/${exp_file}.py" "$SQUAD_DIR"; then
            ((COMPLETED_COUNT++))
        else
            ((FAILED_COUNT++))
            FAILED_EXPERIMENTS+=("$exp_file")
        fi
    done

    # Long SQuAD v2 experiments
    print_header "📊 Long SQuAD v2 Experiments (6-12 segments)"

    for exp_file in "${LONG_SQUAD_EXPERIMENTS[@]}"; do
        # Skip if not in filtered list
        if [ -n "$SPECIFIC_EXPERIMENTS" ]; then
            exp_num=$(echo "$exp_file" | grep -o '^[0-9]\+')
            if ! echo "$SPECIFIC_EXPERIMENTS" | grep -q "$exp_num"; then
                continue
            fi
        fi

        if run_experiment "$LONG_SQUAD_DIR/${exp_file}.py" "$LONG_SQUAD_DIR"; then
            ((COMPLETED_COUNT++))
        else
            ((FAILED_COUNT++))
            FAILED_EXPERIMENTS+=("$exp_file")
        fi
    done

    # Training summary
    print_header "📊 TRAINING SUMMARY"
    echo -e "${GREEN}✅ Completed: $COMPLETED_COUNT${NC}"
    echo -e "${RED}❌ Failed: $FAILED_COUNT${NC}"

    if [ ${#FAILED_EXPERIMENTS[@]} -gt 0 ]; then
        echo ""
        echo -e "${RED}Failed experiments:${NC}"
        for exp in "${FAILED_EXPERIMENTS[@]}"; do
            echo -e "${RED}  - $exp${NC}"
        done
    fi

else
    echo -e "${YELLOW}⏭️  Skipping training phase${NC}"
fi

# ============================================================================
# EVALUATION PHASE
# ============================================================================

if [ "$SKIP_EVALUATION" = false ]; then
    print_header "📊 PHASE 2: EVALUATION"

    echo -e "${BLUE}▶ Evaluating all experiments...${NC}"

    if uv run python "$SCRIPT_DIR/evaluate_all.py" 2>&1 | tee "$LOG_DIR/evaluation.log"; then
        echo -e "${GREEN}✅ Evaluation completed${NC}"
    else
        echo -e "${RED}❌ Evaluation failed${NC}"
        echo -e "${RED}   Check log: $LOG_DIR/evaluation.log${NC}"
    fi

else
    echo -e "${YELLOW}⏭️  Skipping evaluation phase${NC}"
fi

# ============================================================================
# ANALYSIS PHASE
# ============================================================================

if [ "$SKIP_ANALYSIS" = false ]; then
    print_header "📈 PHASE 3: ANALYSIS & VISUALIZATION"

    echo -e "${BLUE}▶ Generating figures...${NC}"

    if uv run python "$SCRIPT_DIR/analyze_results.py" 2>&1 | tee "$LOG_DIR/analysis.log"; then
        echo -e "${GREEN}✅ Analysis completed${NC}"
    else
        echo -e "${RED}❌ Analysis failed${NC}"
        echo -e "${RED}   Check log: $LOG_DIR/analysis.log${NC}"
    fi

else
    echo -e "${YELLOW}⏭️  Skipping analysis phase${NC}"
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

print_header "🎉 ALL PHASES COMPLETE"

echo -e "${GREEN}Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s${NC}"
echo ""
echo -e "${CYAN}📁 Output locations:${NC}"
echo -e "${CYAN}  Training outputs: ./outputs/paper_v2_*/${NC}"
echo -e "${CYAN}  Evaluation results: ./outputs/paper_v2_evaluation_results/${NC}"
echo -e "${CYAN}  Figures: ./outputs/paper_v2_evaluation_results/figures/${NC}"
echo -e "${CYAN}  Logs: $LOG_DIR${NC}"
echo ""
echo -e "${GREEN}✅ Success! Check the outputs directory for results.${NC}"
echo ""
