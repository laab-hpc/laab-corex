CC = nvcc
CFLAGS = -O3

LD_CUBLASMP ?= -lcublasmp -lmpi
LD_NCCL ?= -lnccl
LD_CUBLAS ?= -lcublas
LD_CUDA ?= -lcudart
LD_EXTRA ?=

# MPI_CFLAGS ?= $(shell mpicc --showme:compile 2>/dev/null)
# MPI_LDFLAGS ?= $(shell mpicc --showme:link 2>/dev/null)

LDLIBS = $(LD_CUBLASMP) $(LD_NCCL) $(LD_CUBLAS) $(LD_CUDA) $(LD_EXTRA) -lm

ifndef LAAB_BUILD_DIR
$(error LAAB_BUILD_DIR is not set. Please export it before running make.)
endif

$(shell mkdir -p $(LAAB_BUILD_DIR))

all: pdgemm

pdgemm:
	$(CC) $(CFLAGS) -c pdgemm.cu -o $(LAAB_BUILD_DIR)/pdgemm.o
	$(CC) $(LAAB_BUILD_DIR)/pdgemm.o -o $(LAAB_BUILD_DIR)/pdgemm.exe $(LDLIBS)

.PHONY: all pdgemm clean

clean:
	rm -f $(LAAB_BUILD_DIR)/*.o $(LAAB_BUILD_DIR)/*.exe