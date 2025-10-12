#!/bin/bash
################################################################################
# Run All Paper Experiments
################################################################################
#
# This script runs all 10 paper experiments sequentially, then evaluates
# and analyzes the results.
#
# Usage:
#   ./scripts/run_all_experiments.sh [--quick-test]
#
# Options:
#   --quick-test    Run with small dataset for testing (1000 samples)
#
# Output:
#   - Training results in outputs/paper_exp_*/
#   - Evaluation results in results/paper_experiments/
#   - Figures in results/paper_experiments/figures/
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log file
LOG_FILE="paper_experiments.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "================================================================================
📊 RUNNING ALL PAPER EXPERIMENTS
================================================================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Parse arguments
QUICK_TEST=false
if [[ "$1" == "--quick-test" ]]; then
    QUICK_TEST=true
    echo -e "${YELLOW}⚠️  Quick test mode enabled (small dataset)${NC}"
fi

# Experiments to run
EXPERIMENTS=(
    "01_baseline_no_memory"
    "02_main_memory_8tokens"
    "03_main_memory_16tokens"
    "04_ablation_no_progressive"
    "05_ablation_no_gating"
    "06_ablation_4tokens"
    "07_ablation_32tokens"
    "08_segments_2seg"
    "09_segments_4seg"
    "10_segments_6seg"
)

TOTAL=${#EXPERIMENTS[@]}
COMPLETED=0
FAILED=0

echo ""
echo "Total experiments: ${TOTAL}"
echo "Start time: $(date)"
echo ""

# Run each experiment
for i in "${!EXPERIMENTS[@]}"; do
    EXP="${EXPERIMENTS[$i]}"
    NUM=$((i + 1))

    echo ""
    echo "================================================================================"
    echo -e "${BLUE}📊 EXPERIMENT ${NUM}/${TOTAL}: ${EXP}${NC}"
    echo "================================================================================"
    echo ""

    SCRIPT_PATH="scripts/paper_experiments/${EXP}.py"

    if [[ ! -f "${SCRIPT_PATH}" ]]; then
        echo -e "${RED}❌ Script not found: ${SCRIPT_PATH}${NC}"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Run experiment
    START_TIME=$(date +%s)

    if uv run python "${SCRIPT_PATH}"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        HOURS=$((DURATION / 3600))
        MINS=$(((DURATION % 3600) / 60))

        echo ""
        echo -e "${GREEN}✅ Experiment ${NUM}/${TOTAL} completed successfully${NC}"
        echo "Duration: ${HOURS}h ${MINS}m"
        COMPLETED=$((COMPLETED + 1))
    else
        echo ""
        echo -e "${RED}❌ Experiment ${NUM}/${TOTAL} failed${NC}"
        FAILED=$((FAILED + 1))

        # Ask if user wants to continue
        if [[ "${QUICK_TEST}" != "true" ]]; then
            read -p "Continue with remaining experiments? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Stopping experiments."
                break
            fi
        fi
    fi

    echo ""
    echo "Progress: ${COMPLETED} completed, ${FAILED} failed, $((TOTAL - COMPLETED - FAILED)) remaining"
    echo ""
done

echo ""
echo "================================================================================"
echo -e "${BLUE}📊 TRAINING SUMMARY${NC}"
echo "================================================================================"
echo ""
echo "Total experiments: ${TOTAL}"
echo -e "${GREEN}Completed: ${COMPLETED}${NC}"
echo -e "${RED}Failed: ${FAILED}${NC}"
echo ""

# Run evaluation if any experiments succeeded
if [[ ${COMPLETED} -gt 0 ]]; then
    echo ""
    echo "================================================================================"
    echo -e "${BLUE}📊 EVALUATING ALL EXPERIMENTS${NC}"
    echo "================================================================================"
    echo ""

    if uv run python scripts/paper_experiments/evaluate_all.py; then
        echo -e "${GREEN}✅ Evaluation completed${NC}"

        # Run analysis
        echo ""
        echo "================================================================================"
        echo -e "${BLUE}📊 ANALYZING RESULTS${NC}"
        echo "================================================================================"
        echo ""

        if uv run python scripts/paper_experiments/analyze_results.py; then
            echo -e "${GREEN}✅ Analysis completed${NC}"
        else
            echo -e "${RED}❌ Analysis failed${NC}"
        fi
    else
        echo -e "${RED}❌ Evaluation failed${NC}"
    fi
fi

echo ""
echo "================================================================================"
echo -e "${GREEN}🎉 ALL EXPERIMENTS COMPLETE${NC}"
echo "================================================================================"
echo ""
echo "End time: $(date)"
echo ""
echo "📁 Training outputs: outputs/paper_exp_*/"
echo "📁 Evaluation results: results/paper_experiments/"
echo "📁 Figures: results/paper_experiments/figures/"
echo "📄 Log file: ${LOG_FILE}"
echo ""
echo "================================================================================"
