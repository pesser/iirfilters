#pragma once

#include <vector>
#include <stdexcept>

/**
 * \brief Deriche Gaussian convolution
 * \param c         coefficients precomputed by deriche_precomp()
 * \param dest      output convolved data
 * \param buffer_l    workspace array with space for at least N elements
 * \param buffer_r    workspace array with space for at least N elements
 * \param src       data to be convolved
 * \param N         number of samples
 */
template <class num>
void deriche_seq(
    const deriche_coeffs<num>& c,
    num* dest,
    num* buffer_l,
    num* buffer_r,
    const num* src,
    int N)
{
    assert(dest && buffer_l && buffer_r && src &&
           buffer_l != src && buffer_r != src &&
           N > 0);
    
    if(N <= 4)
    {
        throw std::runtime_error("Not implemented for short inputs (<= 4).");
    }

    /* Initialize boundary values of filter. We simply use zero here but
     * some boundary extension might be desireable. */
    for(int i = 0; i < DERICHE_MAX_K; ++i)
    {
      buffer_l[i] = 0;
      buffer_r[i] = 0;
    }

    switch(c.K)
    {
      case 2:
        for(int n = 2; n < N; ++n)
          buffer_l[n] =
              c.b_causal[0] * src[n]
            + c.b_causal[1] * src[n - 1]
            - c.a[1] * buffer_l[n - 1] 
            - c.a[2] * buffer_l[n - 2];
        break;
      case 3:
        for(int n = 3; n < N; ++n)
          buffer_l[n] =
              c.b_causal[0] * src[n]
            + c.b_causal[1] * src[n - 1]
            + c.b_causal[2] * src[n - 2]
            - c.a[1] * buffer_l[n - 1] 
            - c.a[2] * buffer_l[n - 2]
            - c.a[3] * buffer_l[n - 3];
        break;
      case 4:
        for(int n = 4; n < N; ++n)
          buffer_l[n] =
              c.b_causal[0] * src[n]
            + c.b_causal[1] * src[n - 1]
            + c.b_causal[2] * src[n - 2]
            + c.b_causal[3] * src[n - 3]
            - c.a[1] * buffer_l[n - 1] 
            - c.a[2] * buffer_l[n - 2]
            - c.a[3] * buffer_l[n - 3]
            - c.a[4] * buffer_l[n - 4];
        break;
    }

    /* Why is src[i] and b_anticausal[0] not used? */
    int n, i;
    switch(c.K)
    {
      case 2:
        n = 2;
        i = (N - 1) - n;
        for(; n < N; ++n, --i)
          buffer_r[n] =
              c.b_anticausal[1] * src[i + 1]
            + c.b_anticausal[2] * src[i + 2]
            - c.a[1] * buffer_r[n - 1] 
            - c.a[2] * buffer_r[n - 2];
        break;
      case 3:
        n = 3;
        i = (N - 1) - n;
        for(; n < N; ++n, --i)
          buffer_r[n] =
              c.b_anticausal[1] * src[i + 1]
            + c.b_anticausal[2] * src[i + 2]
            + c.b_anticausal[3] * src[i + 3]
            - c.a[1] * buffer_r[n - 1] 
            - c.a[2] * buffer_r[n - 2] 
            - c.a[3] * buffer_r[n - 3];
        break;
      case 4:
        n = 4;
        i = (N - 1) - n;
        for(; n < N; ++n, --i)
          buffer_r[n] =
              c.b_anticausal[1] * src[i + 1]
            + c.b_anticausal[2] * src[i + 2]
            + c.b_anticausal[3] * src[i + 3]
            + c.b_anticausal[4] * src[i + 4]
            - c.a[1] * buffer_r[n - 1] 
            - c.a[2] * buffer_r[n - 2] 
            - c.a[3] * buffer_r[n - 3]
            - c.a[4] * buffer_r[n - 4];
        break;
    }

    for(int n = 0; n < N; ++n)
      dest[n] = buffer_l[n] + buffer_r[N - 1 - n];

    return;
}

template <class num>
void transpose(num* t, num* src, int width_src, int height_src)
{
  for(int y = 0; y < height_src; ++y)
  {
    for(int x = 0; x < width_src; ++x)
    {
      t[x * height_src + y] = src[y * width_src + x];
    }
  }
}

/**
 * \brief Deriche Gaussian 2D convolution
 * \param c             coefficients precomputed by deriche_precomp()
 * \param dest          output convolved data
 * \param buffer_l      array with at least max(width,height) elements
 * \param buffer_r      array with at least max(width,height) elements
 * \param buffer_m      array with at least width * height elements
 * \param src           data to be convolved
 * \param width         image width
 * \param height        image height
 */
template <class num>
void deriche_seq_2d(
    const deriche_coeffs<num>& c,
    num *dest,
    num *buffer_l,
    num *buffer_r,
    num *buffer_m,
    const num *src, 
    int width, int height)
{
    int num_pixels = width * height;
    
    assert(dest && buffer_l && buffer_r && src && num_pixels > 0);
    
    num* dest_y = dest;
    const num* src_y = src;
    for(int y = 0; y < height; ++y, dest_y += width, src_y += width)
      deriche_seq(c,
                  dest_y,
                  buffer_l,
                  buffer_r,
                  src_y,
                  width);

    transpose(buffer_m, dest, width, height);

    num* dest_x = buffer_m;
    const num* src_x = buffer_m;
    for(int x = 0; x < width; ++x, dest_x += height, src_x += height)
      deriche_seq(c, 
                  dest_x,
                  buffer_l,
                  buffer_r,
                  src_x,
                  height);

    transpose(dest, buffer_m, height, width);
}
