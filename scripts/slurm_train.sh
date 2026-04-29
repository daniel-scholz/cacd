#!/bin/bash
# SLURM training template for CACD.
#
# Usage examples:
#   sbatch scripts/slurm_train.sh scanner_harm --conf <config_name>
#   sbatch --gres=gpu:2 --tasks-per-node=2 scripts/slurm_train.sh scanner_harm --conf <config_name>
#   sbatch --gres=gpu:4 --tasks-per-node=4 scripts/slurm_train.sh scanner_harm --conf <config_name>
#
# For multi-GPU runs, --tasks-per-node must match --gres=gpu:N (Lightning expects
# one task per GPU). Adjust --time / --mem-per-cpu / --cpus-per-task to your cluster.

#SBATCH --job-name=cacd
#SBATCH --output=slurm_out/%x-%A-%t.out
#SBATCH --error=slurm_out/%x-%A-%t.err
#SBATCH --time=11-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G

export PATH="$HOME/.local/bin:$PATH"

# Multi-GPU NCCL hygiene (no-op for single-GPU).
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

mkdir -p slurm_out
nvidia-smi

srun uv run python run.py "$@"
