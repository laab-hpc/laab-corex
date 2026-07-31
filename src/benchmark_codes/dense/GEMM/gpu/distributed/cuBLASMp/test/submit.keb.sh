#!/bin/bash
#SBATCH -A hpc2n2026-184
#SBATCH --job-name=laab-pxgemm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=a100:2
#SBATCH --time=00:20:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --threads-per-core=1
#SBATCH -C a100

set -euo pipefail

export LAAB_BUILD_DIR=$(pwd)/build
export LAAB_TRACE_DIR=$(pwd)/traces
mkdir -p $LAAB_TRACE_DIR

GIT_ROOT=$(git rev-parse --show-toplevel)
INPUTS_DIR=$GIT_ROOT/inputs

module use /proj/nobackup/aravind/software/easybuild-installs/easybuild/modules/all/MPI/GCC/14.3.0/OpenMPI/5.0.8/
module load cuBLASMp
module list

lscpu

make -C ../src clean
make -C ../src

export OMP_NUM_THREADS=1
srun -N 2 -n 4 -c 1 --gpus-per-task=a100:1 --gpu-bind=closest --distribution=block:cyclic:cyclic --threads-per-core=1 "$LAAB_BUILD_DIR/pgemm.exe" -A $INPUTS_DIR/dense/M15000x15000-float64-gen.dense -B $INPUTS_DIR/dense/M15000x15000-float64-gen.dense -b 256 --reps 5 --tag "fp64"
