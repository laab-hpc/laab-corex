# Require LAAB_BUILD_DIR
ifndef LAAB_BUILD_DIR
$(error LAAB_BUILD_DIR is not set. Please export it before running make.)
endif

HIPCC ?= hipcc
HIPFLAGS ?= -O3
LDLIBS ?= -lrocblas

$(shell mkdir -p $(LAAB_BUILD_DIR))

all: gemm dgemm sgemm zgemm

gemm:
	$(HIPCC) $(HIPFLAGS) -c gemm.cpp -o $(LAAB_BUILD_DIR)/gemm_rocblas.o
	$(HIPCC) $(HIPFLAGS) $(LAAB_BUILD_DIR)/gemm_rocblas.o -o $(LAAB_BUILD_DIR)/gemm_rocblas.exe $(LDLIBS)

dgemm: gemm
	$(HIPCC) $(HIPFLAGS) $(LAAB_BUILD_DIR)/gemm_rocblas.o -o $(LAAB_BUILD_DIR)/dgemm_rocblas.exe $(LDLIBS)

sgemm: gemm
	$(HIPCC) $(HIPFLAGS) $(LAAB_BUILD_DIR)/gemm_rocblas.o -o $(LAAB_BUILD_DIR)/sgemm_rocblas.exe $(LDLIBS)

zgemm: gemm
	$(HIPCC) $(HIPFLAGS) $(LAAB_BUILD_DIR)/gemm_rocblas.o -o $(LAAB_BUILD_DIR)/zgemm_rocblas.exe $(LDLIBS)

.PHONY: all gemm dgemm sgemm zgemm clean

clean:
	rm -f $(LAAB_BUILD_DIR)/*.o $(LAAB_BUILD_DIR)/*.exe
