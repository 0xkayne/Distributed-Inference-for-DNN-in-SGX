#!/bin/bash
# Automated script for grouped Inception V3 profiling
# This script automatically updates STORE_CHUNK_ELEM and rebuilds for each group

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$PROJECT_ROOT/Include/common_with_enclaves.h"

cd "$PROJECT_ROOT"

# Group configurations
declare -A GROUP_CHUNK_ELEM=(
    ["Stem"]=130560500
    ["Inception-A"]=940800
    ["Reduction-A"]=134175475
    ["Inception-B"]=221952
    ["Reduction-B"]=1109760
    ["Inception-C"]=30720
    ["Classifier"]=256000
)

# Group order
GROUPS=("Stem" "Inception-A" "Reduction-A" "Inception-B" "Reduction-B" "Inception-C" "Classifier")

echo "=========================================="
echo "Grouped Inception V3 Profiling Script"
echo "=========================================="
echo ""
echo "This script will:"
echo "1. Update STORE_CHUNK_ELEM for each group"
echo "2. Rebuild SGX code"
echo "3. Run profiling for that group"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Backup original config
cp "$CONFIG_FILE" "$CONFIG_FILE.backup"
echo "✓ Backed up original config to $CONFIG_FILE.backup"

# Process each group
for group in "${GROUPS[@]}"; do
    chunk_elem=${GROUP_CHUNK_ELEM[$group]}
    
    echo ""
    echo "=========================================="
    echo "Processing Group: $group"
    echo "STORE_CHUNK_ELEM: $chunk_elem"
    echo "=========================================="
    
    # Update STORE_CHUNK_ELEM
    echo "Updating STORE_CHUNK_ELEM..."
    sed -i "s/#define STORE_CHUNK_ELEM [0-9]*/#define STORE_CHUNK_ELEM $chunk_elem/" "$CONFIG_FILE"
    sed -i "s/#define WORK_CHUNK_ELEM [0-9]*/#define WORK_CHUNK_ELEM $chunk_elem/" "$CONFIG_FILE"
    echo "✓ Updated to $chunk_elem"
    
    # Rebuild
    echo "Rebuilding SGX code..."
    if make clean && make; then
        echo "✓ Build successful"
    else
        echo "✗ Build failed. Please check errors above."
        echo "Restoring original config..."
        cp "$CONFIG_FILE.backup" "$CONFIG_FILE"
        exit 1
    fi
    
    # Run profiling for this group
    echo "Running profiling for $group..."
    python3 -m experiments.models.profile_inception --grouped --input-size 299 || {
        echo "⚠ Warning: Profiling failed for $group, continuing..."
    }
    
    echo "✓ Completed $group"
done

# Restore original config
echo ""
echo "Restoring original config..."
cp "$CONFIG_FILE.backup" "$CONFIG_FILE"
echo "✓ Done!"

echo ""
echo "=========================================="
echo "All groups completed!"
echo "Check inception_metrics.csv for results"
echo "=========================================="



