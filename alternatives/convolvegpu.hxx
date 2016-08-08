/*
 * Header file for convolve_iir_gpu function
 *
 */

#ifndef CONVOLVEGPU_HXX
#define CONVOLVEGPU_HXX

#include "fastfilters.hxx"

void convolve_iir_gpu( const float*, float*, const unsigned int, const unsigned int, const fastfilters::iir::Coefficients & );

#endif
