

CFLAGS=-std=c++11

iirfilter: iirfilter.cxx svenpeter_convolve_iir_nosimd.cxx svenpeter_kernel_iir_deriche.cxx 
	g++ $(CFLAGS) iirfilter.cxx svenpeter_convolve_iir_nosimd.cxx svenpeter_kernel_iir_deriche.cxx -lpng -o iirfilter

iirfilter-cuda: iirfilter.cu
	nvcc iirfilter.cu -o iirfilter-cuda

all:
	iirfilter irrfilter-cuda
