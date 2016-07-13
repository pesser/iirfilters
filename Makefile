

CFLAGS=-std=c++11



iir: iir_sequential.cxx svenpeter_convolve_iir_nosimd.cxx svenpeter_kernel_iir_deriche.cxx 
	g++ $(CFLAGS) iir_sequential.cxx svenpeter_convolve_iir_nosimd.cxx svenpeter_kernel_iir_deriche.cxx -lpng -o iir-sequential

all:
	iir
