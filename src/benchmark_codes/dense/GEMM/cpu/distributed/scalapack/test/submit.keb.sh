#!/bin/bash
#SBATCH -A hpc2n2026-184
#SBATCH --job-name=laab-pdgemm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=14
#SBATCH --time=00:20:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH -C zen4

set -euo pipefail

export LAAB_BUILD_DIR=$(pwd)/build
export LAAB_TRACE_DIR=$(pwd)/traces
mkdir -p $LAAB_TRACE_DIR

GIT_ROOT=$(git rev-parse --show-toplevel)
INPUTS_DIR=$GIT_ROOT/inputs


module load GCC OpenMPI ScaLAPACK

lscpu


make -C ../src clean
make -C ../src


export OMP_NUM_THREADS=14
srun -N 2 -n 4 -c 14 ./build/pdgemm.exe -A $INPUTS_DIR/dense/M15000x15000-float64-gen.dense -B $INPUTS_DIR/dense/M15000x15000-float64-gen.dense -b 256 --reps 5 --tag "fp64"
