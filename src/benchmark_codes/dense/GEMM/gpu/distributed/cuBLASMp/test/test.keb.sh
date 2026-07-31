#get allocation
#salloc -A hpc2n2026-184 -n 4

ml purge
ml use /proj/nobackup/aravind/software/easybuild-installs/easybuild/modules/all/MPI/GCC/14.3.0/OpenMPI/5.0.8/
ml cuBLASMp
ml list

export LAAB_BUILD_DIR=build
make -C ../src
mkdir -p traces

srun -N 2 --ntasks-per-node=2 --gpus-per-task=a100:1 --gpu-bind=closest --distribution=block:cyclic:cyclic --threads-per-core=1 ./build/pgemm.exe -A inputs/dense/M15000x15000-float64-gen.dense -B inputs/dense/M15000x15000-float64-gen.dense -b 256 --reps 5 --tag fp64
