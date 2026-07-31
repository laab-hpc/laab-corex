#!/bin/bash
#SBATCH --job-name=laab-rocblas
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH -A hpc2n2026-184
#SBATCH --gpus=1
#SBATCH --threads-per-core=1
#SBATCH -C amd_gpu
##SBATCH -C nvidia_gpu

set -euo pipefail

export LAAB_BUILD_DIR=$(pwd)/build
export LAAB_TRACE_DIR=$(pwd)/traces
mkdir -p "$LAAB_TRACE_DIR"

GIT_ROOT=$(git rev-parse --show-toplevel)
INPUTS_DIR=${INPUTS_DIR:-$GIT_ROOT/inputs}
REPS="${REPS:-5}"

module load GCC/11.3.0 rocBLAS

lscpu
lspci | grep -Ei 'nvidia|amd|ati|intel|vga|3d|display'

make -C ../src clean
make -C ../src

srun "$LAAB_BUILD_DIR/gemm_rocblas.exe" -A "$INPUTS_DIR/dense/M3000x3000-float64-gen.dense" -B "$INPUTS_DIR/dense/M3000x3000-float64-gen.dense" --reps "$REPS" --tag 0
srun "$LAAB_BUILD_DIR/gemm_rocblas.exe" -A "$INPUTS_DIR/dense/M3000x3000-float32-gen.dense" -B "$INPUTS_DIR/dense/M3000x3000-float32-gen.dense" --reps "$REPS" --tag 1
srun "$LAAB_BUILD_DIR/gemm_rocblas.exe" -A "$INPUTS_DIR/dense/M3000x3000-complex128-gen.dense" -B "$INPUTS_DIR/dense/M3000x3000-complex128-gen.dense" --reps "$REPS" --tag 2
