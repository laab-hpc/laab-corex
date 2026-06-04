module load GCC FlexiBLAS
export LAAB_BUILD_DIR=$(pwd)/build
make -C ../ LD_BLAS=-lopenblas
./build/correctness.exe
./build/dgemm.exe -n 1000 -m 1000 --reps 10
./build/sgemm.exe -n 1000 -m 1000 --reps 10
