CXX ?= g++
NVCC ?= nvcc
CCBIN ?= $(CXX)

CXXFLAGS += -Wall -Wextra -std=c++11 -I. -O3
LDFLAGS += -lpng
NVCCFLAGS += -ccbin=$(CCBIN) -std=c++11 -I. --expt-extended-lambda --compiler-options="-O3" -lcublas

EXECS = example_thrust_deriche example_seq_deriche test_thrust_deriche time_thrust_deriche time_seq_deriche

all: $(EXECS)

debug: CXXFLAGS += -g
debug: NVCCFLAGS += -g
debug: all

omp: NVCCFLAGS += -Xcompiler -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp
omp: all

test_thrust_deriche: test_thrust_deriche.cu
	$(NVCC) $(NVCCFLAGS) $^ -lpng -o $@

example_thrust_deriche: example_thrust_deriche.cu
	$(NVCC) $(NVCCFLAGS) $^ -lpng -o $@

time_thrust_deriche: time_thrust_deriche.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@

time_seq_deriche: time_seq_deriche.cpp

clean:
	rm -f $(EXECS)
