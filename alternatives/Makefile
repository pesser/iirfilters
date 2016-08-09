
NVCCFLAGS=-std=c++11
CFLAGS=-std=c++11


all: iirfilter analyze-iir-thrust analyze-iir-cuda iirfilter-thrust iirfilter-cuda


iirfilter: iirfilter.cxx iir_sequential.o coefs.o
	g++ $(CFLAGS) -I .. -o $@ $^ -lpng

iirfilter-cuda: conv_image.o iir_cuda.o coefs.o
	nvcc $(NVCCFLAGS) -o $@ $^ -lpng 

iirfilter-thrust: conv_image.o iir_thrust.o coefs.o
	nvcc $(NVCCFLAGS) -o $@ $^ -lpng

analyze-iir-cuda: analytics.o iir_sequential.o iir_cuda.o coefs.o
	nvcc $(NVCCFLAGS) -I .. -o $@ $^ -lpng -G -g 

analyze-iir-thrust: analytics.o iir_sequential.o iir_thrust.o coefs.o
	nvcc $(NVCCFLAGS) -I .. -o $@ $^ -lpng


coefs.o: svenpeter_kernel_iir_deriche.cxx
	g++ $(CFLAGS) -c -o $@ $^

iir_sequential.o: svenpeter_convolve_iir_nosimd.cxx
	g++ $(CFLAGS) -c -o $@ $^

conv_image.o: convolve_image.cu
	nvcc -c $(NVCCFLAGS) -I .. -o $@ $^ 

analytics.o: analytics.cu
	nvcc -c $(NVCCFLAGS) -I .. -o $@ $^ 

iir_cuda.o: iirfilter.cu
	nvcc -c $(NVCCFLAGS) -o $@ $^ 

iir_thrust.o: iirfilter_thrust_prefix.cu
	nvcc -c $(NVCCFLAGS) -o $@ $^ 


clean:
	rm -f *.o iirfilter analyze-iir-thrust analyze-iir-cuda iirfilter-thrust iirfilter-cuda

