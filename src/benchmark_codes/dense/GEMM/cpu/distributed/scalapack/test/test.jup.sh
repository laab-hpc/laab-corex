#get allocation
#salloc -A zam -N 2

ml purge
ml GCC OpenMPI ScaLAPACK

export LAAB_BUILD_DIR=build
make -C ../

srun -N 2 --ntasks-per-node=4 --cpus-per-task=72 ./build/pdgemm.exe -A inputs/dense/M15000x15000-float64-gen.dense -B inputs/dense/M15000x15000-float64-gen.dense -b 1000 --reps 5



