#!/bin/bash
# Run full C-TAEA replication with maximum parallelism
# Optimized for 8* GPU, 64-thread system

set -e

echo "==================================================================="
echo "C-TAEA Full Replication Script"
echo "System: 8 * NVIDIA RTX A6000, 64 CPU threads, 1TB RAM"
echo "==================================================================="

# Set environment for optimal NumPy/MKL performance on multi-core
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Prevent Python hash randomization for reproducibility
export PYTHONHASHSEED=42

OUTPUT_DIR="${1:-Results_$(date +%Y%m%d_%H%M%S)}"
WORKERS="${2:-60}"

echo ""
echo "Configuration:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Workers: $WORKERS"
echo "  Runs per config: 51"
echo "  Total configs: 11 problems * 5 m-values * 6 algorithms = 330"
echo "  Total runs: 330 * 51 = 16,830"
echo ""
echo "Estimated time: ~2-4 hours (depending on problem difficulty)"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run with the parallel runner
python -m Experiments.parallel_runner \
    --full \
    --workers "$WORKERS" \
    --output "$OUTPUT_DIR" \
    --batch-size 2

echo ""
echo "==================================================================="
echo "Replication completed!"
echo "Results saved to: $OUTPUT_DIR/"
echo "==================================================================="

# Optional: Run analysis
read -p "Run analysis now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python -m Analysis.analysis --input "$OUTPUT_DIR"
fi
