# cuBLAS GEMM

This implementation mirrors the CPU `cblas` GEMM driver more closely:

- file-based inputs via `-A` and `-B`
- dtype inferred from filenames
- supported precisions: `float32`, `float64`, `complex128`
- CPU-style LAAB trace records with `--tag`

Build:

```bash
export LAAB_BUILD_DIR=$(pwd)/build
make
```

Run:

```bash
export LAAB_TRACE_DIR=$(pwd)/traces
mkdir -p "$LAAB_TRACE_DIR"

$LAAB_BUILD_DIR/gemm_cu.exe \
  -A /path/to/M3000x3000-float64-gen.dense \
  -B /path/to/M3000x3000-float64-gen.dense \
  --reps 5 \
  --tag 0
```

Legacy executable names `dgemm_cu.exe`, `sgemm_cu.exe`, and `zgemm_cu.exe`
are also produced for compatibility. They all dispatch by matrix filename dtype.
