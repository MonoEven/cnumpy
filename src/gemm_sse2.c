/**
 * SSE2 baseline row-tile kernel for contiguous float64 GEMM.
 */
#include <cnumpy/cnumpy_internal.h>
#include <emmintrin.h>

#define CNP_GEMM_I_BLOCK 32
#define CNP_GEMM_K_BLOCK 64
#define CNP_GEMM_J_BLOCK 64

void cnp_sse2_gemm_tile(
    const double *a, const double *b, double *c,
    int64_t m, int64_t n, int64_t k,
    int64_t row_begin, int64_t row_end) {
    (void)m;
    for (int64_t i_block = row_begin; i_block < row_end;
         i_block += CNP_GEMM_I_BLOCK) {
        int64_t i_end = i_block + CNP_GEMM_I_BLOCK;
        if (i_end > row_end) i_end = row_end;
        for (int64_t k_block = 0; k_block < k;
             k_block += CNP_GEMM_K_BLOCK) {
            int64_t k_end = k_block + CNP_GEMM_K_BLOCK;
            if (k_end > k) k_end = k;
            for (int64_t j_block = 0; j_block < n;
                 j_block += CNP_GEMM_J_BLOCK) {
                int64_t j_end = j_block + CNP_GEMM_J_BLOCK;
                if (j_end > n) j_end = n;
                for (int64_t row = i_block; row < i_end; ++row) {
                    const double *a_row = a + row * k;
                    double *c_row = c + row * n;
                    for (int64_t inner = k_block; inner < k_end; ++inner) {
                        const double *b_row = b + inner * n;
                        __m128d a_value = _mm_set1_pd(a_row[inner]);
                        int64_t column = j_block;
                        for (; column + 1 < j_end; column += 2) {
                            __m128d product = _mm_mul_pd(
                                a_value, _mm_loadu_pd(b_row + column));
                            __m128d accumulated = _mm_add_pd(
                                _mm_loadu_pd(c_row + column), product);
                            _mm_storeu_pd(c_row + column, accumulated);
                        }
                        for (; column < j_end; ++column)
                            c_row[column] += a_row[inner] * b_row[column];
                    }
                }
            }
        }
    }
}
