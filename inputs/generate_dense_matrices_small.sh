mgen --outdir dense/ --prec float64 --shape 3000x3000 --workers 12
mgen --outdir dense/ --prec float64 --shape 6000x100 --workers 12
mgen --outdir dense/ --prec float64 --shape 100x100 --workers 12
mgen --outdir dense/ --prec float64 --shape 100x6000 --workers 12

mgen --outdir dense/ --prec float32 --shape 3000x3000 --workers 12
mgen --outdir dense/ --prec float32 --shape 6000x100 --workers 12
mgen --outdir dense/ --prec float32 --shape 100x100 --workers 12
mgen --outdir dense/ --prec float32 --shape 100x6000 --workers 12

mgen --outdir dense/ --prec complex128 --shape 3000x3000 --workers 12
mgen --outdir dense/ --prec complex128 --shape 6000x100 --workers 12
mgen --outdir dense/ --prec complex128 --shape 100x100 --workers 12
mgen --outdir dense/ --prec complex128 --shape 100x6000 --workers 12
