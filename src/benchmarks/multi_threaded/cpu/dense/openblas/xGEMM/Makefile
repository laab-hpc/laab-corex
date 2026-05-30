LD_BLAS = -lopenblas

# Require LAAB_BUILD_DIR
ifndef LAAB_BUILD_DIR
$(error LAAB_BUILD_DIR is not set. Please export it before running make.)
endif


# Ensure build dir exists
$(shell mkdir -p $(LAAB_BUILD_DIR))

all: dgemm sgemm correctness

dgemm:
	gcc -O3 -c dgemm.c -o $(LAAB_BUILD_DIR)/dgemm.o
	gcc $(LAAB_BUILD_DIR)/dgemm.o -o $(LAAB_BUILD_DIR)/dgemm.exe $(LD_BLAS)

sgemm:
	gcc -O3 -c sgemm.c -o $(LAAB_BUILD_DIR)/sgemm.o
	gcc $(LAAB_BUILD_DIR)/sgemm.o -o $(LAAB_BUILD_DIR)/sgemm.exe $(LD_BLAS)

correctness:
	gcc -O3 -c correctness.c -o $(LAAB_BUILD_DIR)/correctness.o
	gcc $(LAAB_BUILD_DIR)/correctness.o -o $(LAAB_BUILD_DIR)/correctness.exe $(LD_BLAS)


.PHONY: all dgemm sgemm correctness clean

clean:
	rm -f $(LAAB_BUILD_DIR)/*.o $(LAAB_BUILD_DIR)/*.exe \
