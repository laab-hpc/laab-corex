#!/bin/bash
#SBATCH --job-name=laab-openblas
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
##SBATCH -C skylake&nvidia_gpu

set -euo pipefail


export LAAB_BUILD_DIR=$(pwd)/build
export LAAB_TRACE_DIR=$(pwd)/traces
mkdir -p $LAAB_TRACE_DIR

module load GCC FlexiBLAS

lscpu

make -C ../ clean
make -C ../ LD_BLAS=-lopenblas

export OMP_NUM_THREADS=24
srun -c 24 "$LAAB_BUILD_DIR/gemm.exe" -A ../inputs/M3000x3000-float64-gen.dense -B ../inputs/M3000x3000-float64-gen.dense --reps 5 --tag 0
srun -c 24 "$LAAB_BUILD_DIR/gemm.exe" -A ../inputs/M3000x3000-float32-gen.dense -B ../inputs/M3000x3000-float32-gen.dense --reps 5 --tag 1
srun -c 24 "$LAAB_BUILD_DIR/gemm.exe" -A ../inputs/M3000x3000-complex128-gen.dense -B ../inputs/M3000x3000-complex128-gen.dense --reps 5 --tag 2


