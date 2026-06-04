#!/bin/bash
#SBATCH -A hpc2n2026-184
#SBATCH --job-name=laab-pdgemm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:20:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err

set -euo pipefail

export LAAB_BUILD_DIR=$(pwd)/build

module load GCC OpenMPI ScaLAPACK

lscpu

M="${M:-3000}"
N="${N:-3000}"
B="${B:-256}"
REPS="${REPS:-5}"
NP="${NP:-1}"

make -C ../ clean
make -C ../

#srun -n 4 $LAAB_BUILD_DIR/correctness.exe

export OMP_NUM_THREADS=24
srun -n "$NP" -c 24 "$LAAB_BUILD_DIR/pdgemm.exe" -m "$M" -n "$N" -b "$B" --reps "$REPS"
