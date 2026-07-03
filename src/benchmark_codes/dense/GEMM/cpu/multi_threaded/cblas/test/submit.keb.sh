#!/bin/bash
#SBATCH --job-name=laab-openblas
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --threads-per-core=1
#SBATCH --time=00:20:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH -A hpc2n2026-184
##SBATCH -C skylake

set -euo pipefail


export LAAB_BUILD_DIR=$(pwd)/build
export LAAB_TRACE_DIR=$(pwd)/traces
mkdir -p $LAAB_TRACE_DIR

GIT_ROOT=$(git rev-parse --show-toplevel)
INPUTS_DIR=$GIT_ROOT/inputs

module load GCC FlexiBLAS

lscpu

make -C ../src clean
make -C ../src LD_BLAS=-lopenblas

export OMP_NUM_THREADS=14
srun -c 14 "$LAAB_BUILD_DIR/gemm.exe" -A $INPUTS_DIR/dense/M3000x3000-float64-gen.dense -B $INPUTS_DIR/dense/M3000x3000-float64-gen.dense --reps 5 --tag 0
srun -c 14 "$LAAB_BUILD_DIR/gemm.exe" -A $INPUTS_DIR/dense/M3000x3000-float32-gen.dense -B $INPUTS_DIR/dense/M3000x3000-float32-gen.dense --reps 5 --tag 1
srun -c 14 "$LAAB_BUILD_DIR/gemm.exe" -A $INPUTS_DIR/dense/M3000x3000-complex128-gen.dense -B $INPUTS_DIR/dense/M3000x3000-complex128-gen.dense --reps 5 --tag 2


