#!/bin/bash
#SBATCH --job-name=linx-openblas
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --cpus-per-task=24
#SBATCH -A hpc2n2026-184
#SBATCH --gpus=1
#SBATCH --threads-per-core=1
#SBATCH -C skylake&nvidia_gpu

set -euo pipefail


export LAAB_BUILD_DIR=$(pwd)/build

module load GCC OpenBLAS FlexiBLAS

REPS="${REPS:-5}"

NT=1
export OMP_NUM_THREADS=$NT

lscpu

make -C ../ clean
make -C ../ LD_BLAS=-lopenblas
srun $LAAB_BUILD_DIR/correctness.exe

srun -c $NT "$LAAB_BUILD_DIR/dgemm.exe" -m 3000 -n 3000 --reps "$REPS"
srun -c $NT "$LAAB_BUILD_DIR/sgemm.exe" -m 3000 -n 3000 --reps "$REPS"


