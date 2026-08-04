/**
 * AVX2 row-tile kernel for contiguous float64 GEMM.
 * Compiled in an isolated /arch:AVX2, non-LTCG translation unit.
 */
#include <cnumpy/cnumpy_internal.h>
#include <immintrin.h>

#define CNP_GEMM_I_BLOCK 32
#define CNP_GEMM_K_BLOCK 64
#define CNP_GEMM_J_BLOCK 128

__declspec(noinline) void cnp_avx2_gemm_tile(
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
                        __m256d a_value = _mm256_set1_pd(a_row[inner]);
                        int64_t column = j_block;
                        for (; column + 3 < j_end; column += 4) {
                            __m256d product = _mm256_mul_pd(
                                a_value, _mm256_loadu_pd(b_row + column));
                            __m256d accumulated = _mm256_add_pd(
                                _mm256_loadu_pd(c_row + column), product);
                            _mm256_storeu_pd(c_row + column, accumulated);
                        }
                        for (; column < j_end; ++column)
                            c_row[column] += a_row[inner] * b_row[column];
                    }
                }
            }
        }
    }
}
