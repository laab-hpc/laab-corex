#!/bin/bash
#SBATCH -A hpc2n2026-184
#SBATCH --job-name=laab-pdgemm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=20
#SBATCH --time=00:20:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err

set -euo pipefail

ROOT_DIR=/proj/nobackup/aravind/LAAB/benchmarks/laab-dense/src/multinode-cpu/scalapack/pxGEMM
cd "$ROOT_DIR"

export LAAB_BUILD_DIR="${LAAB_BUILD_DIR:-$ROOT_DIR/build}"

module load GCC OpenMPI ScaLAPACK

lscpu

M="${M:-10000}"
N="${N:-10000}"
B="${B:-256}"
REPS="${REPS:-5}"
NP="${NP:-40}"

make clean
make

srun -n "$NP" "$LAAB_BUILD_DIR/pdgemm.exe" -m "$M" -n "$N" -b "$B" --reps "$REPS"
