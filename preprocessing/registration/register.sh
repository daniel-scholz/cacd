#!/bin/bash
# Rigidly affine-register every NIfTI in a directory to an MNI152 atlas with
# NiftyReg's reg_aladin (https://github.com/KCL-BMEIS/niftyreg).
#
# Usage: ./register.sh <input_dir> <output_dir> <atlas.nii.gz>
set -euo pipefail

dataIn=${1:?"input directory required"}
dataOut=${2:?"output directory required"}
atlas=${3:?"atlas file required (e.g. MNI152 1mm T1)"}

reg_aladin_bin=${REG_ALADIN:-reg_aladin}
mkdir -p "$dataOut"

for file in "$dataIn"/*.nii.gz; do
    name=$(basename "$file")
    out="$dataOut/$name"
    if [ -f "$out" ]; then
        echo "$out exists, skipping"
        continue
    fi
    "$reg_aladin_bin" -speeeeed -flo "$file" -ref "$atlas" -res "$out" -rigOnly
done
