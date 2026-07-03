LD_BLAS = -lopenblas

# Require LAAB_BUILD_DIR
ifndef LAAB_BUILD_DIR
$(error LAAB_BUILD_DIR is not set. Please export it before running make.)
endif


# Ensure build dir exists
$(shell mkdir -p $(LAAB_BUILD_DIR))

all: gemm

gemm:
	gcc -O3 -c gemm.c -o $(LAAB_BUILD_DIR)/gemm.o
	gcc $(LAAB_BUILD_DIR)/gemm.o -o $(LAAB_BUILD_DIR)/gemm.exe $(LD_BLAS)

gemm_intel:
	icx -DUSE_MKL -c gemm.c -o $(LAAB_BUILD_DIR)/gemm.o
	icx $(LAAB_BUILD_DIR)/gemm.o -o $(LAAB_BUILD_DIR)/gemm.exe -qmkl


.PHONY: all gemm clean

clean:
	rm -f $(LAAB_BUILD_DIR)/*.o $(LAAB_BUILD_DIR)/*.exe \
