#!/bin/bash
#SBATCH -A hpc2n2026-184
#SBATCH --job-name=laab-pdgemm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err

set -euo pipefail

export LAAB_BUILD_DIR=$(pwd)/build

module load GCC OpenMPI ScaLAPACK

lscpu


make -C ../ clean
make -C ../


MATRIX_DIR=../practise/matrices/dense
export LAAB_TRACE_DIR=$(pwd)/traces
srun -N 2 -n 16 ./build/pdgemm.exe -A $MATRIX_DIR/M15000x15000-float64-gen.dense -B $MATRIX_DIR/M15000x15000-float64-gen.dense -b 256 --reps 5
