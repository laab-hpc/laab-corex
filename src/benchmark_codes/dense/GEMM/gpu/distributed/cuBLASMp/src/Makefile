CC = nvcc
CFLAGS = -O3 -Wno-deprecated-gpu-targets

LD_CUBLASMP ?= -lcublasmp -lmpi
LD_NCCL ?= -lnccl
LD_CUBLAS ?= -lcublas
LD_CUDA ?= -lcudart
LD_EXTRA ?=
LDLIBS = $(LD_CUBLASMP) $(LD_NCCL) $(LD_CUBLAS) $(LD_CUDA) $(LD_EXTRA) -lm

ifndef LAAB_BUILD_DIR
$(error LAAB_BUILD_DIR is not set. Please export it before running make.)
endif

$(shell mkdir -p $(LAAB_BUILD_DIR))

all: pgemm pdgemm psgemm pzgemm

pgemm:
	$(CC) $(CFLAGS) -c pgemm.cu -o $(LAAB_BUILD_DIR)/pgemm.o
	$(CC) $(LAAB_BUILD_DIR)/pgemm.o -o $(LAAB_BUILD_DIR)/pgemm.exe $(LDLIBS)

pdgemm: pgemm
	$(CC) $(LAAB_BUILD_DIR)/pgemm.o -o $(LAAB_BUILD_DIR)/pdgemm.exe $(LDLIBS)

psgemm: pgemm
	$(CC) $(LAAB_BUILD_DIR)/pgemm.o -o $(LAAB_BUILD_DIR)/psgemm.exe $(LDLIBS)

pzgemm: pgemm
	$(CC) $(LAAB_BUILD_DIR)/pgemm.o -o $(LAAB_BUILD_DIR)/pzgemm.exe $(LDLIBS)

.PHONY: all pgemm pdgemm psgemm pzgemm clean

clean:
	rm -f $(LAAB_BUILD_DIR)/*.o $(LAAB_BUILD_DIR)/*.exe
