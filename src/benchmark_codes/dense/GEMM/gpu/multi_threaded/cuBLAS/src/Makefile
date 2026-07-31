# Require LAAB_BUILD_DIR
ifndef LAAB_BUILD_DIR
$(error LAAB_BUILD_DIR is not set. Please export it before running make.)
endif

NVCC ?= nvcc
NVCCFLAGS ?= -O3
LDLIBS ?= -lcublas

$(shell mkdir -p $(LAAB_BUILD_DIR))

all: gemm dgemm sgemm zgemm

gemm:
	$(NVCC) $(NVCCFLAGS) -c gemm.cu -o $(LAAB_BUILD_DIR)/gemm_cu.o
	$(NVCC) $(NVCCFLAGS) $(LAAB_BUILD_DIR)/gemm_cu.o -o $(LAAB_BUILD_DIR)/gemm_cu.exe $(LDLIBS)

dgemm: gemm
	$(NVCC) $(NVCCFLAGS) $(LAAB_BUILD_DIR)/gemm_cu.o -o $(LAAB_BUILD_DIR)/dgemm_cu.exe $(LDLIBS)

sgemm: gemm
	$(NVCC) $(NVCCFLAGS) $(LAAB_BUILD_DIR)/gemm_cu.o -o $(LAAB_BUILD_DIR)/sgemm_cu.exe $(LDLIBS)

zgemm: gemm
	$(NVCC) $(NVCCFLAGS) $(LAAB_BUILD_DIR)/gemm_cu.o -o $(LAAB_BUILD_DIR)/zgemm_cu.exe $(LDLIBS)

.PHONY: all gemm dgemm sgemm zgemm clean

clean:
	rm -f $(LAAB_BUILD_DIR)/*.o $(LAAB_BUILD_DIR)/*.exe
