#!/bin/bash
#SBATCH -A zam
#SBATCH --job-name=laab-pxgemm
#SBATCH --nodes=2
#SBATCH --time=00:20:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err

set -euo pipefail

export LAAB_BUILD_DIR=$(pwd)/build
export LAAB_TRACE_DIR=$(pwd)/traces
mkdir -p $LAAB_TRACE_DIR

GIT_ROOT=$(git rev-parse --show-toplevel)
INPUTS_DIR=$GIT_ROOT/inputs

module purge
module load cuBLASMp
module list

lscpu

make -C ../src clean
make -C ../src

export OMP_NUM_THREADS=1
srun -N 2 --ntasks-per-node=4 --cpus-per-task=72 --gpus-per-task=1 "$LAAB_BUILD_DIR/pgemm.exe" -A $INPUTS_DIR/dense/M15000x15000-float64-gen.dense -B $INPUTS_DIR/dense/M15000x15000-float64-gen.dense -b 256 --reps 5 --tag "fp64"
