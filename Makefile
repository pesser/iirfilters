CXX ?= g++
NVCC ?= nvcc
CCBIN ?= $(CXX)

CXXFLAGS += -Wall -Wextra -std=c++11 -I.
LDFLAGS += -lpng
NVCCFLAGS += -ccbin=$(CCBIN) -std=c++11 -I.

EXECS = iirfilter test_coefficients test_seq_deriche test_seq_deriche_2d

all: $(EXECS)

iirfilter: iirfilter.cxx svenpeter_convolve_iir_nosimd.cxx svenpeter_kernel_iir_deriche.cxx 
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

iirfilter-cuda: iirfilter.cu
	$(NVCC) $(NVCCFLAGS) iirfilter.cu -o iirfilter-cuda

clean:
	rm -f $(EXECS)
