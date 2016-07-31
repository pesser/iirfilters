

CFLAGS=-std=c++11

iirfilter: iirfilter.cxx svenpeter_convolve_iir_nosimd.cxx svenpeter_kernel_iir_deriche.cxx fastfilters.hxx image.hxx
	g++ $(CFLAGS) iirfilter.cxx svenpeter_convolve_iir_nosimd.cxx svenpeter_kernel_iir_deriche.cxx -lpng -o iirfilter

iirfilter-cuda: iirfilter.cu svenpeter_kernel_iir_deriche.cxx fastfilters.hxx image.hxx
	nvcc $(CFLAGS) iirfilter.cu svenpeter_kernel_iir_deriche.cxx -lpng -o iirfilter-cuda

iirfilter-thrust: iirfilter_thrust.cu svenpeter_kernel_iir_deriche.cxx fastfilters.hxx image.hxx
	nvcc $(CFLAGS) iirfilter_thrust.cu svenpeter_kernel_iir_deriche.cxx -lpng -o iirfilter-thrust

all:
	iirfilter iirfilter-cuda iirfilter-thrust
