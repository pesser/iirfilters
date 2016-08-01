/*
  convolve_iir_gpu function for convolving an image horizontally using pure cuda

*/
#include "fastfilters.hxx"

#define N_REGISTERS_CONVOLVE_ROW 40 // determine with -Xptxas -v compiler flags


__global__ void convolve_row( float* input, float* output, int M, int N, int n_border, 
			      float* d, float* causal, float* anticausal )
{
  int dim = blockIdx.x*blockDim.x + threadIdx.x;
  if (dim < M) { 
     // original row code 
  }
}


void convolve_iir_gpu( const float* input, float* output, const unsigned int n_cols, const unsigned int n_rows,
                       const fastfilters::iir::Coefficients &coefs)
{
  cudaSetDevice(0);
  float *in, *out, *d, *causal, *anticausal; 
  unsigned int n_blocks, n_threads_per_block, coefs_size, data_size;  
  cudaDeviceProp prop;

  data_size = n_cols * n_rows * sizeof(float);
  coefs_size = 4 * sizeof(float);
  
  cudaMalloc( &in, data_size );
  cudaMalloc( &out, data_size );
  cudaMalloc( &d, coefs_size );
  cudaMalloc( &causal, coefs_size );
  cudaMalloc( &anticausal, coefs_size );

  cudaMemcpy( in, input, data_size, cudaMemcpyHostToDevice );

  cudaMemcpy( d, coefs.d.data(), coefs_size, cudaMemcpyHostToDevice );
  cudaMemcpy( causal, coefs.n_causal.data(), coefs_size, cudaMemcpyHostToDevice );
  cudaMemcpy( anticausal, coefs.n_anticausal.data(), coefs_size, cudaMemcpyHostToDevice );

  cudaGetDeviceProperties( &prop, 0 );
  n_blocks = ceil( N_REGISTERS_CONVOLVE_ROW * n_rows / (float) prop.regsPerBlock ); 
  n_threads_per_block = ceil( n_rows/ (float) n_blocks );
  
  convolve_row<<< n_blocks, n_threads_per_block >>>
    ( in, out, n_rows, n_cols, coefs.n_border, d, causal, anticausal );

  cudaMemcpy( output, out, data_size, cudaMemcpyDeviceToHost );
  
  cudaFree( in );
  cudaFree( out );
  cudaFree( d );
  cudaFree( causal );
  cudaFree( anticausal );
}

