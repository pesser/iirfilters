## IIR Filter alternative implementations

Includes the original sequential algorithm, a naive CUDA implementation based this sequential version and a Thrust implementation using segmented prefix sums. Since the latter, parallel implementations are of exploratory character, they only implement the horizontal direction.

### Usage
This section includes executables to compare the CUDA implementation and the prefix sum implementation directly with the orignal algorithm and infrastructure to compare them with the Thrust-foreach implementation and the sequential implementation used as reference to evaluate its speedup.

### Executables

- `iirfilter`: Performs horizontal Gaussian blurring on a given image using the
  original implementation.
- `iirfilter_cuda`: Same as above with naive CUDA implementation.
- `iirfilter_thrust_prefix`: Same as above with Thrust prefix sum implementation.
- `analyze_cuda`: Performs IIR filter on square data with specified
  number of rows and prints three columns:
  1. Number of rows
  2. Time for horizontal pass CUDA version with data transfers forth and back
  3. Time for horizontal pass of original algorithm
- `analyze_thrust_prefix`: Same as above with Thrust prefix sum implementation.

### Other Files

- `svenpeter_convolve_iir_nosimd.cxx`: Original sequential algorithm.
- `svenpeter_kernel_iir_deriche.cxx`: Original code for computing deriche coefficients.
- `fastfilters.hxx`: Header file for original code.
- `convolve_image.cu`: Used for iirfilter_cuda and iirfilter_thrust_prefix to convolve a given image. 
- `analytics.cu`: Used for analyze_cuda and analyze_thrust_prefix to compare performance of parallel implementation with original code. 
- `convolvegpu.hxx`: Header file for convolve_image.cu and analytics.cu. 
- `analyze_cuda.sh`: Script to time CUDA convolutions.
- `analyze_thrust_prefix.sh`: Script to time Thrust prefix sum convolutions.
- `analyze_thrust_foreach.sh`: Script to time Thrust foreach convolutions.
- `analyze_thrust_seq.sh`: Script to time sequential reference implementation for Thrust-foreach version.
- `performance.py`: Collects results from these four bash scripts and computes comparisons. (Time for original code taken from results of analyze_cuda.sh)

