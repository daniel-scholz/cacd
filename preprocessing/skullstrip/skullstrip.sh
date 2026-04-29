#!/bin/bash
# Skull-strip every NIfTI in a directory with HD-BET (https://github.com/MIC-DKFZ/HD-BET).
#
# Usage: ./skullstrip.sh <input_dir> <output_dir>
set -euo pipefail

dataIn=${1:?"input directory required"}
dataOut=${2:?"output directory required"}

mkdir -p "$dataOut"

for file in "$dataIn"/*.nii.gz; do
    name=$(basename "$file")
    out="$dataOut/$name"
    if [ -f "$out" ]; then
        echo "$out exists, skipping"
        continue
    fi
    hd-bet -i "$file" -o "$out"
done
