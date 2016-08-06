#pragma once

#include <vector>
#include <stdexcept>

/**
 * \brief Deriche Gaussian 2D convolution. Only for coefficients of order 4
 * and data with more than 5 rows and columns.
 * \param c               coefficients precomputed by deriche_precomp()
 * \param dest            output convolved data
 * \param buffer_l        array with at least max(width,height) elements
 * \param buffer_r        array with at least max(width,height) elements
 * \param src             data to be convolved
 * \param height          image height
 * \param width           image width
 * \param row_stride      stride along dimension 0 (i.e. &src(i + 1, j) - &src(i, j) = row_stride)
 * \param column_stride   stride along dimension 1 (i.e. &src(i, j + 1) - &src(i, j) = column_stride)
 */
template <class num>
void deriche_seq_2d(
    const deriche_coeffs<num>& c,
    num *dest,
    num *buffer_l,
    num *buffer_r,
    const num *src, 
    int height, int width,
    int row_stride, int column_stride)
{
    assert(dest && buffer_l && buffer_r && src && c.K == 4 && width > 4 && height > 4);
    const int b_column_stride = 1;
    
    num* dest_y = dest;
    const num* src_y = src;
    for(int y = 0; y < height; ++y, dest_y += row_stride, src_y += row_stride)
    {
      // causal
      // init
      for(int i = 0; i < 4; ++i)
      {
        buffer_l[i * b_column_stride] = 0;
        for(int j = 0; j <= i; ++j)
          buffer_l[i * b_column_stride] += c.b_causal[j] * src_y[(i - j) * column_stride];
        for(int j = 1; j <= i; ++j)
          buffer_l[i * b_column_stride] -= c.a[j] * buffer_l[(i - j) * b_column_stride];
      }
      // compute
      for(int n = 4; n < width; ++n)
        buffer_l[n * b_column_stride] =
            c.b_causal[0] * src_y[(n - 0) * column_stride]
          + c.b_causal[1] * src_y[(n - 1) * column_stride]
          + c.b_causal[2] * src_y[(n - 2) * column_stride]
          + c.b_causal[3] * src_y[(n - 3) * column_stride]
          - c.a[1] * buffer_l[(n - 1) * b_column_stride] 
          - c.a[2] * buffer_l[(n - 2) * b_column_stride]
          - c.a[3] * buffer_l[(n - 3) * b_column_stride]
          - c.a[4] * buffer_l[(n - 4) * b_column_stride];
      // anticausal
      // init
      for(int i = 0; i < 4; ++i)
      {
        buffer_r[i * b_column_stride] = 0;
        for(int j = 1; j <= i; ++j)
          buffer_r[i * b_column_stride] += c.b_anticausal[j] * src_y[((width - 1) - i + j) * column_stride];
        for(int j = 1; j <= i; ++j)
          buffer_r[i * b_column_stride] -= c.a[j] * buffer_r[(i - j) * b_column_stride];
      }
      // compute
      int n, i;
      n = 4;
      i = (width - 1) - n;
      for(; n < width; ++n, --i)
        buffer_r[n * b_column_stride] =
            c.b_anticausal[1] * src_y[(i + 1) * column_stride]
          + c.b_anticausal[2] * src_y[(i + 2) * column_stride]
          + c.b_anticausal[3] * src_y[(i + 3) * column_stride]
          + c.b_anticausal[4] * src_y[(i + 4) * column_stride]
          - c.a[1] * buffer_r[(n - 1) * b_column_stride] 
          - c.a[2] * buffer_r[(n - 2) * b_column_stride] 
          - c.a[3] * buffer_r[(n - 3) * b_column_stride]
          - c.a[4] * buffer_r[(n - 4) * b_column_stride];

      // combine
      for(int n = 0; n < width; ++n)
        dest_y[n * column_stride] = buffer_l[n * b_column_stride] + buffer_r[(width - 1 - n) * b_column_stride];
    }
}
