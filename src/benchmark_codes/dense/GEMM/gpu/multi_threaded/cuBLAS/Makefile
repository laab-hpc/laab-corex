LD_BLAS = -lopenblas

# Require LAAB_BUILD_DIR
ifndef LAAB_BUILD_DIR
$(error LAAB_BUILD_DIR is not set. Please export it before running make.)
endif


# Ensure build dir exists
$(shell mkdir -p $(LAAB_BUILD_DIR))

all: dgemm sgemm correctness

dgemm:
	nvcc -c dgemm.cu -o $(LAAB_BUILD_DIR)/dgemm_cu.o
	nvcc $(LAAB_BUILD_DIR)/dgemm_cu.o -o $(LAAB_BUILD_DIR)/dgemm_cu.exe -lcublas

sgemm:
	nvcc -c sgemm.cu -o $(LAAB_BUILD_DIR)/sgemm_cu.o
	nvcc $(LAAB_BUILD_DIR)/sgemm_cu.o -o $(LAAB_BUILD_DIR)/sgemm_cu.exe -lcublas

correctness:
	nvcc -c correctness.cu -o $(LAAB_BUILD_DIR)/correctness_cu.o
	nvcc $(LAAB_BUILD_DIR)/correctness_cu.o -o $(LAAB_BUILD_DIR)/correctness_cu.exe -lcublas


.PHONY: all dgemm sgemm correctness clean

clean:
	rm -f $(LAAB_BUILD_DIR)/*.o $(LAAB_BUILD_DIR)/*.exe \
