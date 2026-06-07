#!/bin/bash
#SBATCH --job-name=cublas-dgemm
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
#SBATCH -C amd_gpu
##SBATCH -C nvidia_gpu

set -euo pipefail


export LAAB_BUILD_DIR=$(pwd)/build
export LAAB_TRACE_DIR=$(pwd)/traces

module load GCC/11.3.0 rocBLAS

REPS="${REPS:-5}"


lscpu
lspci | grep -Ei 'nvidia|amd|ati|intel|vga|3d|display'

make -C ../ clean
make -C ../ 

srun $LAAB_BUILD_DIR/correctness_rocblas.exe
srun "$LAAB_BUILD_DIR/dgemm_rocblas.exe" -m 30000 -n 30000 -k 3000 --reps "$REPS"
srun "$LAAB_BUILD_DIR/sgemm_rocblas.exe" -m 30000 -n 30000 -k 3000 --reps "$REPS"


