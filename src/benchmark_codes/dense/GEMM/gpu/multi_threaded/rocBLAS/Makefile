# Require LAAB_BUILD_DIR
ifndef LAAB_BUILD_DIR
$(error LAAB_BUILD_DIR is not set. Please export it before running make.)
endif

$(shell mkdir -p $(LAAB_BUILD_DIR))

all: dgemm sgemm correctness

dgemm:
	hipcc -O3 -c dgemm.cpp -o $(LAAB_BUILD_DIR)/dgemm_rocblas.o
	hipcc $(LAAB_BUILD_DIR)/dgemm_rocblas.o -o $(LAAB_BUILD_DIR)/dgemm_rocblas.exe -lrocblas

sgemm:
	hipcc -O3 -c sgemm.cpp -o $(LAAB_BUILD_DIR)/sgemm_rocblas.o
	hipcc $(LAAB_BUILD_DIR)/sgemm_rocblas.o -o $(LAAB_BUILD_DIR)/sgemm_rocblas.exe -lrocblas

correctness:
	hipcc -O3 -c correctness.cpp -o $(LAAB_BUILD_DIR)/correctness_rocblas.o
	hipcc $(LAAB_BUILD_DIR)/correctness_rocblas.o -o $(LAAB_BUILD_DIR)/correctness_rocblas.exe -lrocblas

.PHONY: all dgemm sgemm correctness clean

clean:
	rm -f $(LAAB_BUILD_DIR)/*.o $(LAAB_BUILD_DIR)/*.exe
