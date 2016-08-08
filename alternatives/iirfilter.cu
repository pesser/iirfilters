/*
  convolve_iir_gpu function for convolving an image horizontally using pure cuda

*/
#include "fastfilters.hxx"

#define N_REGISTERS_CONVOLVE_ROW 40 // determine with -Xptxas -v compiler flags


__global__ void convolve_row( float* input, float* output, int M, int N, int n_border, 
			      float* d, float* causal, float* anticausal )
{
  int dim = blockIdx.x*blockDim.x + threadIdx.x;
  if (dim < M){ 
  const float *cur_line = input + dim * N;
  float *cur_output = output + dim * N;
  float xtmp[4];
  float ytmp[4];
  float* tmpbfr = new float[N];
  for (unsigned int i = 0; i < 4; ++i)
    xtmp[i] = ytmp[i] = 0.0;

  // left border
  for (unsigned int i = 0; i < n_border; ++i) {
    float sum = 0.0;

    xtmp[0] = cur_line[n_border - i];
    for (unsigned int j = 0; j < 4; ++j)
      sum += causal[j] * xtmp[j];
    for (unsigned int j = 0; j < 4; ++j)
      sum -= d[j] * ytmp[j];
    for (unsigned int j = 3; j > 0; --j) {
      xtmp[j] = xtmp[j - 1];
      ytmp[j] = ytmp[j - 1];
    }

    ytmp[0] = sum;
  }

  // causal pass
  for (unsigned int i = 0; i < N; ++i) {
    float sum = 0.0;

    xtmp[0] = cur_line[i];
    for (unsigned int j = 0; j < 4; ++j)
      sum += causal[j] * xtmp[j];
    for (unsigned int j = 0; j < 4; ++j)
      sum -= d[j] * ytmp[j];
    for (unsigned int j = 3; j > 0; --j) {
      xtmp[j] = xtmp[j - 1];
      ytmp[j] = ytmp[j - 1];
    }

    tmpbfr[i] = sum;
    ytmp[0] = sum;
  }

  // reset variables for anti-causal pass
  for (unsigned int i = 0; i < 4; ++i)
    xtmp[i] = ytmp[i] = 0.0;

  // right border
  for (int i = n_border; i > 0; --i) {
    float sum = 0.0;

    for (unsigned int j = 0; j < 4; ++j)
      sum += anticausal[j] * xtmp[j];
    for (unsigned int j = 0; j < 4; ++j)
      sum -= d[j] * ytmp[j];
    for (unsigned int j = 3; j > 0; --j) {
      xtmp[j] = xtmp[j - 1];
      ytmp[j] = ytmp[j - 1];
    }

    xtmp[0] = cur_line[N - i];
    ytmp[0] = sum;
  }

  // anti-causal pass
  for (int i = N - 1; i >= 0; --i) {
    float sum = 0.0;

    for (unsigned int j = 0; j < 4; ++j)
      sum += anticausal[j] * xtmp[j];
    for (unsigned int j = 0; j < 4; ++j)
      sum -= d[j] * ytmp[j];
    for (unsigned int j = 3; j > 0; --j) {
      xtmp[j] = xtmp[j - 1];
      ytmp[j] = ytmp[j - 1];
    }

    xtmp[0] = cur_line[i];
    ytmp[0] = sum;
    cur_output[i] = tmpbfr[i] + sum;
  }
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

