
NVCCFLAGS=-std=c++11
CFLAGS=-std=c++11

EXECS = iirfilter iirfilter_cuda iirfilter_thrust_prefix analyze_cuda analyze_thrust_prefix

all: $(EXECS)


iirfilter: iirfilter.cxx iir_sequential.o coefs.o
	g++ $(CFLAGS) -I .. -o $@ $^ -lpng

iirfilter_cuda: conv_image.o iirfilter_cuda.o coefs.o
	nvcc $(NVCCFLAGS) -o $@ $^ -lpng 

iirfilter_thrust_prefix: conv_image.o iirfilter_thrust_prefix.o coefs.o
	nvcc $(NVCCFLAGS) -o $@ $^ -lpng

analyze_cuda: analytics.o iirfilter.o iirfilter_cuda.o coefs.o
	nvcc $(NVCCFLAGS) -I .. -o $@ $^ -lpng

analyze_thrust_prefix: analytics.o iirfilter.o iirfilter_thrust_prefix.o coefs.o
	nvcc $(NVCCFLAGS) -I .. -o $@ $^ -lpng


coefs.o: svenpeter_kernel_iir_deriche.cxx
	g++ $(CFLAGS) -c -o $@ $^

iirfilter.o: svenpeter_convolve_iir_nosimd.cxx
	g++ $(CFLAGS) -c -o $@ $^

conv_image.o: convolve_image.cu
	nvcc -c $(NVCCFLAGS) -o $@ $^ 

analytics.o: analytics.cu
	nvcc -c $(NVCCFLAGS) -o $@ $^ 

iirfilter_cuda.o: iirfilter_cuda.cu
	nvcc -c $(NVCCFLAGS) -o $@ $^ 

iirfilter_thrust_prefix.o: iirfilter_thrust_prefix.cu
	nvcc -c $(NVCCFLAGS) -o $@ $^ 


clean:
	rm -f *.o $(EXECS)

