module load GCC FlexiBLAS
export LAAB_BUILD_DIR=$(pwd)/build
export LAAB_TRACE_DIR=$(pwd)/traces
mkdir -p $LAAB_TRACE_DIR
make -C ../ LD_BLAS=-lopenblas
./build/gemm.exe -A ../inputs/M3000x3000-float64-gen.dense -B ../inputs/M3000x3000-float64-gen.dense --reps 5 --tag 0
./build/gemm.exe -A ../inputs/M3000x3000-float32-gen.dense -B ../inputs/M3000x3000-float32-gen.dense --reps 5 --tag 1
export OMP_NUM_THREADS=4
./build/gemm.exe -A ../inputs/M3000x3000-float64-gen.dense -B ../inputs/M3000x3000-float64-gen.dense --reps 5 --tag 0
./build/gemm.exe -A ../inputs/M3000x3000-float32-gen.dense -B ../inputs/M3000x3000-float32-gen.dense --reps 5 --tag 1
