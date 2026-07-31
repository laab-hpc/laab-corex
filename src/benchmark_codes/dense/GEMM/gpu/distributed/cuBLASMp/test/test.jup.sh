#get allocation
#salloc -A zam -N 2

ml purge
ml cuBLASMp
ml list

export LAAB_BUILD_DIR=build
make -C ../src
mkdir -p traces

srun -N 2 --ntasks-per-node=4 --gpus-per-task=1 ./build/pgemm.exe -A inputs/dense/M15000x15000-float64-gen.dense -B inputs/dense/M15000x15000-float64-gen.dense -b 256 --reps 5 --tag fp64
