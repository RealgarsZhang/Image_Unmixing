#v. of Tennessee, Koxville
#//////////////////////////////////////////////////////////////////////////////

CC        = icc 
CXX       = icpc
FORT      = ifort

ARCH      = ar
ARCHFLAGS = cr
RANLIB    = ranlib


# --------------------
#  # flags

# --------------------
#  # libraries

# see MKL Link Advisor at http://software.intel.com/sites/products/mkl/
# # icc/ifort with MKL 10.3, Intel OpenMP threads
#
 LIBMKL   = -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lcublas -lcudart -lstdc++ -lm -qopenmp 
# -lcublas ,-lcudart -qopenmp are added by me.
 LIBCUDA  = -L/opt/packages/cuda/8.0/lib64 -lcublas -lcusparse -lcudart -lcudadevrt
#
 LIBMAGMA = -L/home/xihearch/git/magma/lib -lmagma -lmagma_sparse
#
#
# # --------------------
# # directories
#
 INCCUDA   = -I/opt/packages/cuda/8.0/include
#
 INCMKL  = -I${MKLROOT}/include
#
 INCMAGMA = -I/home/xihearch/git/magma/include
#
NoNoiseNoPrintMagma.o: NoNoiseNoPrintMagma.c NoNoiseNoPrintMagma.h	
	$(CC) -c NoNoiseNoPrintMagma.c -DADD_ $(INCMKL) $(INCCUDA) $(INCMAGMA)

dlatanh.o: dlatanh.cu 
	nvcc -O3 -DNDEBUG -DADD_ -Xcompiler "-fPIC -Wall -Wno-unused-function"  -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61 $(INCCUDA) $(INCMAGMA) -c -o dlatanh.o dlatanh.cu

dla_pmult_dtanh.o: dla_pmult_dtanh.cu
	nvcc -O3 -DNDEBUG -DADD_ -Xcompiler "-fPIC -Wall -Wno-unused-function"  -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61 $(INCCUDA) $(INCMAGMA) -c -o dla_pmult_dtanh.o dla_pmult_dtanh.cu

all: mainfMagma.c NoNoiseNoPrintMagma.o dlatanh.o dla_pmult_dtanh.o
	$(CC) -c mainfMagma.c -DADD_ $(INCMKL) $(INCCUDA) $(INCMAGMA)
	$(CC) -o exe mainfMagma.o NoNoiseNoPrintMagma.o $(LIBMKL) $(LIBCUDA) $(LIBMAGMA) dla_pmult_dtanh.o dlatanh.o

NoNoiseNoPrintCPU.o: NoNoiseNoPrintCPU.c NoNoiseNoPrintCPU.h
	$(CC) -c NoNoiseNoPrintCPU.c -mkl

CPUall: mainfCPU.c NoNoiseNoPrintCPU.o
	$(CC) -c mainfCPU.c -mkl
	$(CC) -o exe mainfCPU.o NoNoiseNoPrintCPU.o -mkl

clean : 
	rm ./*.o                  
