module purge
module load GCC/14.3.0 Python/3.13.5
source /proj/nobackup/aravind/LAAB/inspectors/laab_inspector/venv-prod/bin/activate

laab-process traces/ \
    --name "dense/GEMM/cpu/distributed" \
    --system kebnekaise:zen4 \
    --lib scalapack \
    --version 2.2.2 \
    --toolchain foss2024a \
    --np 4 \
    --work_dist "-N 2 -tpN 2 -cpt 14" \
    --cb_id 0 \
    --profile_dir "profiles/"

laab-collect-profiles .

# visualize results
#laab-dashboard .
