# GEMM

### Generate Input Matrices

```
mgen --outdir inputs/ --prec float64 --shape 3000x3000 --workers 12
mgen --outdir inputs/ --prec float32 --shape 3000x3000 --workers 12
mgen --outdir inputs/ --prec complex128 --shape 3000x3000 --workers 12
```
