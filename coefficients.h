#pragma once

#include <complex>
#include <cassert>

/* Pretty much the same as https://github.com/thouis/fastblur.git with some
 * adaptions to use C++ and replacing custom complex numbers with the
 * implementation in the standard library. Useage:
 *
 *  float sigma = 1.0;
 *  int order = 4;
 *  deriche_coeffs<float> c;
 *  deriche_precomp<float>(&c, sigma, order);
 * */

/** \brief Minimum Deriche filter order */
#define DERICHE_MIN_K       2
/** \brief Maximum Deriche filter order */
#define DERICHE_MAX_K       4
/** \brief Test whether a given K value is a valid Deriche filter order */
#define DERICHE_VALID_K(K)  (DERICHE_MIN_K <= (K) && (K) <= DERICHE_MAX_K)
/** \brief The constant sqrt(2 pi) */
#define M_SQRT2PI   2.50662827463100050241576528481104525

// forward declaration
template <class num>
void make_filter(num *result_b,
                 num *result_a, 
                 const std::complex<num> *alpha,
                 const std::complex<num> *beta,
                 int K,
                 double sigma);

/** \brief Coefficients for Deriche Gaussian approximation.
 * Notice that b_anticausal and a start at 1 to resemble the indices of the
 * source factor. */
template <class num>
struct deriche_coeffs
{
    num a[DERICHE_MAX_K + 1];             /**< Denominator coeffs          */
    num b_causal[DERICHE_MAX_K];          /**< Causal numerator            */
    num b_anticausal[DERICHE_MAX_K + 1];  /**< Anticausal numerator        */
    num sum_causal;                       /**< Causal filter sum           */
    num sum_anticausal;                   /**< Anticausal filter sum       */
    num sigma;                            /**< Gaussian standard deviation */
    int K;                                /**< Filter order = 2, 3, or 4   */
};

/**
 * \brief Precompute coefficients for Deriche's Gaussian approximation
 * \param c         deriche_coeffs pointer to hold precomputed coefficients
 * \param sigma     Gaussian standard deviation
 * \param K         filter order = 2, 3, or 4
 * 
 * This routine precomputes the recursive filter coefficients for
 * Deriche's Gaussian convolution approximation.
 */
template <class num>
void deriche_precomp(deriche_coeffs<num> *c, double sigma, int K)
{
    static const std::complex<num> alpha[DERICHE_MAX_K - DERICHE_MIN_K + 1][4] = {
        {{0.48145, 0.971}, {0.48145, -0.971}},
        {{-0.44645, 0.5105}, {-0.44645, -0.5105}, {1.898, 0}},
        {{0.84, 1.8675}, {0.84, -1.8675}, 
            {-0.34015, -0.1299}, {-0.34015, 0.1299}}
        };
    static const std::complex<num> lambda[DERICHE_MAX_K - DERICHE_MIN_K + 1][4] = {
        {{1.26, 0.8448}, {1.26, -0.8448}},
        {{1.512, 1.475}, {1.512, -1.475}, {1.556, 0}},
        {{1.783, 0.6318}, {1.783, -0.6318}, 
            {1.723, 1.997}, {1.723, -1.997}}
        };
    std::complex<num> beta[DERICHE_MAX_K];
    
    int k;
    double accum, accum_denom;
    
    assert(c && sigma > 0 && DERICHE_VALID_K(K));
    
    for(k = 0; k < K; k++)
    {
        double temp = exp(-lambda[K - DERICHE_MIN_K][k].real() / sigma);
        beta[k] = std::complex<num>(
            -temp * cos(lambda[K - DERICHE_MIN_K][k].imag() / sigma),
            temp * sin(lambda[K - DERICHE_MIN_K][k].imag() / sigma));
    }
    
    /* Compute the causal filter coefficients */
    make_filter<num>(c->b_causal, c->a, alpha[K - DERICHE_MIN_K], beta, K, sigma);
    
    /* Numerator coefficients of the anticausal filter */
    c->b_anticausal[0] = (num)(0.0);
    
    for(k = 1; k < K; k++)
        c->b_anticausal[k] = c->b_causal[k] - c->a[k] * c->b_causal[0];
    
    c->b_anticausal[K] = -c->a[K] * c->b_causal[0];
    
    /* Impulse response sums */
    for(k = 1, accum_denom = 1.0; k <= K; k++)
        accum_denom += c->a[k];
    
    for(k = 0, accum = 0.0; k < K; k++)
        accum += c->b_causal[k];
    
    c->sum_causal = (num)(accum / accum_denom);
    
    for(k = 1, accum = 0.0; k <= K; k++)
        accum += c->b_anticausal[k];
    
    c->sum_anticausal = (num)(accum / accum_denom);
    
    c->sigma = (num)sigma;
    c->K = K;    
}

/**
 * \brief Make Deriche filter from alpha and beta coefficients 
 * \param result_b      resulting numerator filter coefficients
 * \param result_a      resulting denominator filter coefficients
 * \param alpha, beta   input coefficients
 * \param K             number of terms
 * \param sigma         Gaussian sigma parameter
 * \ingroup deriche_gaussian
 * 
 * This routine performs the algebraic rearrangement 
 * \f[ \sum_{k=0}^{K-1}\frac{\alpha_k}{1+\beta_k z^{-1}}=\frac{1}{\sqrt{2\pi
\sigma^2}}\frac{\sum_{k=0}^{K-1}b_k z^{-k}}{1+\sum_{k=1}^{K}a_k z^{-k}} \f]
 * to obtain the numerator and denominator coefficients for the causal filter
 * in Deriche's Gaussian approximation.
 * 
 * The routine initializes b/a as the 0th term,
 * \f[ \frac{b(z)}{a(z)} = \frac{\alpha_0}{1 + \beta_0 z^{-1}}, \f]
 * then the kth term is added according to
 * \f[ \frac{b(z)}{a(z)}\leftarrow\frac{b(z)}{a(z)}+\frac{\alpha_k}{1+\beta_k
z^{-1}}=\frac{b(z)(1+\beta_kz^{-1})+a(z)\alpha_k}{a(z)(1+\beta_kz^{-1})}. \f]
 */
template <class num>
void make_filter(num *result_b, num *result_a, 
    const std::complex<num> *alpha, const std::complex<num> *beta, int K, double sigma)
{
    const double denom = sigma * M_SQRT2PI;
    std::complex<num> b[DERICHE_MAX_K], a[DERICHE_MAX_K + 1];
    int k, j;
        
    b[0] = alpha[0];    /* Initialize b/a = alpha[0] / (1 + beta[0] z^-1) */
    a[0] = std::complex<num>(1, 0);
    a[1] = beta[0];
    
    for(k = 1; k < K; k++)
    {   /* Add kth term, b/a += alpha[k] / (1 + beta[k] z^-1) */
        b[k] = beta[k] * b[k-1];
        
        for(j = k - 1; j > 0; j--)
            b[j] = b[j] + (beta[k] * b[j - 1]);
        
        for(j = 0; j <= k; j++)
            b[j] = b[j] + (alpha[k] * a[j]);
        
        a[k + 1] = beta[k] * a[k];
        
        for(j = k; j > 0; j--)
            a[j] = a[j] + (beta[k] * a[j - 1]);
    }
    
    for(k = 0; k < K; k++)
    {
        result_b[k] = (num)(b[k].real() / denom);
        result_a[k + 1] = (num)a[k + 1].real();
    }
    
    return;
}
