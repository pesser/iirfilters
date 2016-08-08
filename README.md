## IIR Filter implementation

Implementation of IIR Filter with [thrust](https://thrust.github.io/).
Calling `make` builds two examples (image blurring), a test and two programs
for timing purposes. By default `thrust`'s `CUDA` backend is used. `make
omp` builds everything with the `OpenMP` backend.
