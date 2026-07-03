module load GCC FlexiBLAS

# Use this to generate input matrices
# mgen --outdir ../inputs/dense/ --prec float64 --shape 3000x3000 --workers 12
# mgen --outdir ../inputs/dense/ --prec float32 --shape 3000x3000 --workers 12
# INPUTS_DIR=../inputs

# Coment out the next two lines if you have generated input matrices
GIT_ROOT=$(git rev-parse --show-toplevel)
INPUTS_DIR=$GIT_ROOT/inputs

export LAAB_BUILD_DIR=$(pwd)/build
export LAAB_TRACE_DIR=$(pwd)/traces

mkdir -p $LAAB_TRACE_DIR
make -C ../src LD_BLAS=-lopenblas

export OMP_NUM_THREADS=1
./build/gemm.exe -A $INPUTS_DIR/dense/M3000x3000-float64-gen.dense -B $INPUTS_DIR/dense/M3000x3000-float64-gen.dense --reps 5 --tag 0
./build/gemm.exe -A $INPUTS_DIR/dense/M3000x3000-float32-gen.dense -B $INPUTS_DIR/dense/M3000x3000-float32-gen.dense --reps 5 --tag 1

export OMP_NUM_THREADS=4
./build/gemm.exe -A $INPUTS_DIR/dense/M3000x3000-float64-gen.dense -B $INPUTS_DIR/dense/M3000x3000-float64-gen.dense --reps 5 --tag 0
./build/gemm.exe -A $INPUTS_DIR/dense/M3000x3000-float32-gen.dense -B $INPUTS_DIR/dense/M3000x3000-float32-gen.dense --reps 5 --tag 1
