

NVCCFLAGS=-std=c++11
CFLAGS=-std=c++11


iirfilter: iirfilter.cxx iir_sequential.o coefs.o fastfilters.hxx image.hxx
	g++ $(CFLAGS) -o $@ $^ -lpng

coefs.o: svenpeter_kernel_iir_deriche.cxx
	g++ $(CFLAGS) -c -o $@ $^

iir_sequential.o: svenpeter_convolve_iir_nosimd.cxx
	g++ $(CFLAGS) -c -o $@ $^

ifeq ("x","y")
iirfilter-cuda: conv_image.o iir_cuda.o coefs.o fastfilters.hxx image.hxx
	nvcc $(NVCCFLAGS) -o $@ $^ -lpng 

iirfilter-thrust: conv_image.o iir_thrust.o coefs.o fastfilters.hxx image.hxx
	nvcc $(NVCCFLAGS) -o $@ $^ -lpng

analyze-iir-cuda: analytics.o iir_cuda.o coefs.o fastfilters.hxx image.hxx
	nvcc $(NVCCFLAGS) -o $@ $^ -lpng 

analyze-iir-thrust: analytics.o iir_thrust.o coefs.o fastfilters.hxx image.hxx
	nvcc $(NVCCFLAGS) -o $@ $^ -lpng

conv_image.o: convolve_image.cxx
	nvcc -c $(NVCCFLAGS) -o $@ $^ 

conv_image.o: analytics.cxx
	nvcc -c $(NVCCFLAGS) -o $@ $^ 

iir_cuda.o: iirfilter.cu
	nvcc -c $(NVCCFLAGS) -o $@ $^ 

iir_thrust.o: iirfilter_thrust.cu
	nvcc -c $(NVCCFLAGS) -o $@ $^ 
endif


all:
	iirfilter

clean:
	rm -f iirfilter *.o

#iirfilter-cuda iirfilter-thrust
