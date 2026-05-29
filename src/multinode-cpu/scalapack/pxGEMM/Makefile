CC = mpicc
CFLAGS = -O3

# Override these from command line/environment if your cluster uses different names.
LD_SCALAPACK ?= -lscalapack
LD_BLACS ?=
LD_BLAS ?= -lopenblas
LD_EXTRA ?= 
LDLIBS = $(LD_SCALAPACK) $(LD_BLACS) $(LD_BLAS) $(LD_EXTRA) -lm

# Require LAAB_BUILD_DIR
ifndef LAAB_BUILD_DIR
$(error LAAB_BUILD_DIR is not set. Please export it before running make.)
endif

# Ensure build dir exists
$(shell mkdir -p $(LAAB_BUILD_DIR))

all: pdgemm

pdgemm:
	$(CC) $(CFLAGS) -c pdgemm.c -o $(LAAB_BUILD_DIR)/pdgemm.o
	$(CC) $(LAAB_BUILD_DIR)/pdgemm.o -o $(LAAB_BUILD_DIR)/pdgemm.exe $(LDLIBS)

.PHONY: all pdgemm clean

clean:
	rm -f $(LAAB_BUILD_DIR)/pdgemm.o $(LAAB_BUILD_DIR)/pdgemm.exe
