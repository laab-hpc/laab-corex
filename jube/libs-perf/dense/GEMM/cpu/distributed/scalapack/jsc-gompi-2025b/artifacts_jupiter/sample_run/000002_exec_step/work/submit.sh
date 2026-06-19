#!/bin/bash
#SBATCH --job-name=distributed_gemm
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=00:20:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH -A zam
#SBATCH --partition booster

set -euo pipefail


SRC_DIR=/e/project1/cjsc/sankaran2/llview-apps/benchmarks/laab-core/src/benchmark_codes/dense/GEMM/cpu/distributed/scalapack
DATA_DIR=/e/project1/cjsc/sankaran2/llview-apps/benchmarks/laab-core/benchmarks/data


module purge
module load Stages/2026
module load GCC OpenMPI ScaLAPACK

set -euo pipefail


lscpu > cpu_info.txt 
export LAAB_BUILD_DIR=$(pwd)/build
export LAAB_TRACE_DIR=$(pwd)/traces
mkdir -p $LAAB_TRACE_DIR


make -C $SRC_DIR clean
make -C $SRC_DIR LD_SCALAPACK=-lscalapack LD_BLAS=-lopenblas


MATRIX_DIR=$SRC_DIR/matrices/dense

srun -N 4 --ntasks-per-node 4  -c 72 ./build/pdgemm.exe -A $MATRIX_DIR/M15000x15000-float64-gen.dense -B $MATRIX_DIR/M15000x15000-float64-gen.dense -b 256 --reps 5
