module load GCC/11.3.0 rocBLAS

# Use this to generate input matrices
# mgen --outdir ../inputs/dense/ --prec float64 --shape 3000x3000 --workers 12
# mgen --outdir ../inputs/dense/ --prec float32 --shape 3000x3000 --workers 12
# mgen --outdir ../inputs/dense/ --prec complex128 --shape 3000x3000 --workers 12

GIT_ROOT=$(git rev-parse --show-toplevel)
INPUTS_DIR=${INPUTS_DIR:-$GIT_ROOT/inputs}

export LAAB_BUILD_DIR=$(pwd)/build
export LAAB_TRACE_DIR=$(pwd)/traces

mkdir -p "$LAAB_TRACE_DIR"
make -C ../src

./build/gemm_rocblas.exe -A "$INPUTS_DIR/dense/M3000x3000-float64-gen.dense" -B "$INPUTS_DIR/dense/M3000x3000-float64-gen.dense" --reps 5 --tag 0
./build/gemm_rocblas.exe -A "$INPUTS_DIR/dense/M3000x3000-float32-gen.dense" -B "$INPUTS_DIR/dense/M3000x3000-float32-gen.dense" --reps 5 --tag 1
./build/gemm_rocblas.exe -A "$INPUTS_DIR/dense/M3000x3000-complex128-gen.dense" -B "$INPUTS_DIR/dense/M3000x3000-complex128-gen.dense" --reps 5 --tag 2
