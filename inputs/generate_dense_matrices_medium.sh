mgen --outdir dense/ --prec float64 --shape 100000x15000 --workers 12
mgen --outdir dense/ --prec float64 --shape 15000x15000 --workers 12
mgen --outdir dense/ --prec float64 --shape 10000x10000 --workers 12
mgen --outdir dense/ --prec float64 --shape 10000x100 --workers 12

mgen --outdir dense/ --prec complex128 --shape 10000x10000 --workers 12
mgen --outdir dense/ --prec complex128 --shape 10000x1000 --workers 12
mgen --outdir dense/ --prec complex128 --shape 10000x100 --workers 12

mgen --outdir dense/ --prec float32 --shape 100000x15000 --workers 12
mgen --outdir dense/ --prec float32 --shape 15000x15000 --workers 12
mgen --outdir dense/ --prec float32 --shape 10000x10000 --workers 12
mgen --outdir dense/ --prec float32 --shape 10000x100 --workers 12
