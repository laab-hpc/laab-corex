# cuBLASMp Distributed GEMM

This `src/` layout mirrors the CPU distributed `scalapack/src` structure more
closely while supporting three precisions from the same driver:

- `float32`
- `float64`
- `complex128`

The program is file-driven and expects matrix filenames like:

```text
M15000x15000-float64-...dense
```

Build:

```bash
export LAAB_BUILD_DIR=$(pwd)/build
make
```

Run:

```bash
export LAAB_TRACE_DIR=$(pwd)/traces
mpirun -np 4 $LAAB_BUILD_DIR/pgemm.exe \
  -A /path/to/M15000x15000-float64-gen.dense \
  -B /path/to/M15000x15000-float64-gen.dense \
  -b 256 \
  --reps 5 \
  --tag fp64
```

Compatibility aliases `pdgemm.exe`, `psgemm.exe`, and `pzgemm.exe` are also
produced, but dtype selection still comes from the matrix filenames.
