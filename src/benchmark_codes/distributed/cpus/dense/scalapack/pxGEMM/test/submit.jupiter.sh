#!/bin/bash
#SBATCH -A admin
#SBATCH --partition booster
#SBATCH --job-name=laab-pdgemm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=288
#SBATCH --time=00:20:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err

set -euo pipefail

export LAAB_BUILD_DIR=$(pwd)/build

module load GCC OpenMPI ScaLAPACK
export SLURM_MPI_TYPE=pspmix

lscpu

M="${M:-10000}"
N="${N:-10000}"
B="${B:-256}"
REPS="${REPS:-10}"
NP="${NP:-20}"

make -C ../ clean
make -C ../

srun -n 4 $LAAB_BUILD_DIR/correctness.exe
export OMP_NUM_THREADS=1
srun -n "$NP" "$LAAB_BUILD_DIR/pdgemm.exe" -m "$M" -n "$N" -b "$B" --reps "$REPS"