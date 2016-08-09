## IIR Filter implementation

Implementation of IIR Filter with [thrust](https://thrust.github.io/). The
parallelization enables fully coalesced memory access by transposing the
data during the addition of the causal and anti-causal pass with
[cuBLAS](https://developer.nvidia.com/cublas). Furthermore, the
implementation can also be used without `CUDA` enabled devices using the
[OpenMP backend of
thrust](https://github.com/thrust/thrust/wiki/Device-Backends). A comparison
between different parallelization approaches can be found in the
presentation slides (and the different approaches' implementations can be
found in the commit history).

### Useage
The repository contains two examples, one test and two programs for timing
purposes. Calling `make` builds all of them with the `CUDA` backend and
`make omp` with the `OpenMP` backend (make sure to call `make clean` if you
want to __re__build).

### Executables

- `example_thrust_deriche`: Performs Gaussian image blurring using the
  thrust implementation.
- `example_seq_deriche`: Same as above with sequential reference
  implementation.
- `test_thrust_deriche`: Compares the results of Gaussian image blurring
  between the thrust implementation and the sequential implementation and
  gives a warning if the error on any pixel is larger than 1e-4.
- `time_thrust_deriche`: Performs IIR filter on square data with specified
  number of rows and prints four columns:
  1. Number of rows
  2. Time for setup (memory allocation and host to device transfer)
  3. Time for horizontal pass
  4. Time for vertical pass
  5. Time for finalizing (device to host transfer)
  Notice that the final version does not use a horizontal pass and columns 3
  and 4 should simply be added together and treated as the compute time.
- `time_seq_deriche`: Same as above with sequential implementation.

### Other Files

- `coefficients.h`: Datastructure and computation of Deriche coefficients
  used to approximate Gaussian blurring with IIR filter.
- `seq_deriche.h`: Sequential implementation.
- `thrust_deriche.h`: Thrust implementation.
- `timer.h`: Timer that can be used with and without CUDA, makes sure to
  synchronize with CUDA and returns time in seconds.
- `utils.h`: Utilities used for debugging and testing.
- `alternatives/`: More alternatives for parallelization to be explored.
- `images/`: Some example images to be used with the examples.
- `png++/`: [PNG++](http://savannah.nongnu.org/projects/pngpp/) - a C++
  interface to `libpng` to read and write `png` images.
- `presentation/`: Presentation slides of this work.

### Contact

- Judith Massa <j.massa@mail.de>
- Patrick Esser <p.esser@stud.uni-heidelberg.de>
