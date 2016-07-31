CXX ?= g++
NVCC ?= nvcc
CCBIN ?= $(CXX)

CXXFLAGS += -Wall -Wextra -std=c++11 -I.
LDFLAGS += -lpng
NVCCFLAGS += -ccbin=$(CCBIN) -std=c++11 -I. --expt-extended-lambda

EXECS = iirfilter test_coefficients test_seq_deriche test_seq_deriche_2d test_thrust_deriche_2d

all: $(EXECS)

debug: CXXFLAGS += -g
debug: NVCCFLAGS += -g
debug: all

iirfilter: iirfilter.cxx svenpeter_convolve_iir_nosimd.cxx svenpeter_kernel_iir_deriche.cxx 
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

iirfilter-cuda: iirfilter.cu
	$(NVCC) $(NVCCFLAGS) iirfilter.cu -o iirfilter-cuda

test_thrust_deriche_2d: test_thrust_deriche_2d.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@

clean:
	rm -f $(EXECS)
