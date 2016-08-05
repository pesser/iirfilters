CXX ?= g++
NVCC ?= nvcc
CCBIN ?= $(CXX)

CXXFLAGS += -Wall -Wextra -std=c++11 -I. -O3
LDFLAGS += -lpng
NVCCFLAGS += -ccbin=$(CCBIN) -std=c++11 -I. --expt-extended-lambda --compiler-options="-O3"

EXECS = iirfilter test_coefficients test_thrust_deriche_2d example_thrust_deriche time_thrust_deriche time_seq_deriche
#test_seq_deriche example_seq_deriche 

all: $(EXECS)

debug: CXXFLAGS += -g
debug: NVCCFLAGS += -g
debug: all

cublas: NVCCFLAGS += -D_CUBLAS_TRANSPOSE
cublas: time_thrust_deriche

omp: NVCCFLAGS += -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp -g
omp: time_thrust_deriche

iirfilter: iirfilter.cxx svenpeter_convolve_iir_nosimd.cxx svenpeter_kernel_iir_deriche.cxx 
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

iirfilter-cuda: iirfilter.cu
	$(NVCC) $(NVCCFLAGS) iirfilter.cu -o iirfilter-cuda

test_thrust_deriche_2d: test_thrust_deriche_2d.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@

example_thrust_deriche: example_thrust_deriche.cu
	$(NVCC) $(NVCCFLAGS) $^ -lpng -o $@

time_thrust_deriche: time_thrust_deriche.cu
	$(NVCC) $(NVCCFLAGS) $^ -lcublas -o $@

clean:
	rm -f $(EXECS)
