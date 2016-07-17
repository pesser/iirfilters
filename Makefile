

CFLAGS=-std=c++11

iirfilter: iirfilter.cxx svenpeter_convolve_iir_nosimd.cxx svenpeter_kernel_iir_deriche.cxx fastfilters.hxx image.hxx
	g++ $(CFLAGS) iirfilter.cxx svenpeter_convolve_iir_nosimd.cxx svenpeter_kernel_iir_deriche.cxx -lpng -o iirfilter

iirfilter-cuda: iirfilter.cu svenpeter_kernel_iir_deriche.cxx fastfilters.hxx image.hxx
	nvcc $(CFLAGS) iirfilter.cu -lpng -o iirfilter-cuda

all:
	iirfilter iirfilter-cuda 
