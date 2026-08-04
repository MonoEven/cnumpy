/**
 * AVX2-only reduction kernels.
 *
 * This translation unit is compiled separately with /arch:AVX2 and without
 * whole-program optimization. Callers must pass through simd_dispatch.c.
 */
#include <cnumpy/cnumpy_internal.h>
#include <immintrin.h>

static __forceinline void cnp_avx2_sin_f32_block(
    const float *source, float *destination) {
    const __m256 zero = _mm256_setzero_ps();
    const __m256 two_over_pi = _mm256_set1_ps(0x1.45f306p-1f);
    const __m256 pio2_high = _mm256_set1_ps(-0x1.921fb0p+0f);
    const __m256 pio2_medium = _mm256_set1_ps(-0x1.5110b4p-22f);
    const __m256 pio2_low = _mm256_set1_ps(-0x1.846988p-48f);
    const __m256 rounding_magic = _mm256_set1_ps(0x1.8p+23f);
    const __m256 maximum_cody = _mm256_set1_ps(117435.992f);
    const __m256i magnitude_mask = _mm256_set1_epi32(0x7fffffff);
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i two = _mm256_set1_epi32(2);

    __m256 input = _mm256_loadu_ps(source);
    __m256 magnitude = _mm256_castsi256_ps(_mm256_and_si256(
        _mm256_castps_si256(input), magnitude_mask));
    __m256 range_mask = _mm256_cmp_ps(
        magnitude, maximum_cody, _CMP_LE_OQ);
    int range_bits = _mm256_movemask_ps(range_mask);
    __m256 reduced_input = _mm256_blendv_ps(zero, input, range_mask);

    __m256 quadrant = _mm256_mul_ps(reduced_input, two_over_pi);
    quadrant = _mm256_add_ps(quadrant, rounding_magic);
    quadrant = _mm256_sub_ps(quadrant, rounding_magic);

    __m256 reduced = _mm256_fmadd_ps(quadrant, pio2_high, reduced_input);
    reduced = _mm256_fmadd_ps(quadrant, pio2_medium, reduced);
    reduced = _mm256_fmadd_ps(quadrant, pio2_low, reduced);
    __m256 reduced_squared = _mm256_mul_ps(reduced, reduced);

    __m256 sine = _mm256_fmadd_ps(
        _mm256_set1_ps(0x1.7d3bbcp-19f), reduced_squared,
        _mm256_set1_ps(-0x1.a06bbap-13f));
    sine = _mm256_fmadd_ps(
        sine, reduced_squared, _mm256_set1_ps(0x1.11119ap-7f));
    sine = _mm256_fmadd_ps(
        sine, reduced_squared, _mm256_set1_ps(-0x1.555556p-3f));
    sine = _mm256_fmadd_ps(sine, reduced_squared, zero);
    sine = _mm256_fmadd_ps(sine, reduced, reduced);

    __m256 cosine = _mm256_fmadd_ps(
        _mm256_set1_ps(0x1.98e616p-16f), reduced_squared,
        _mm256_set1_ps(-0x1.6c06dcp-10f));
    cosine = _mm256_fmadd_ps(
        cosine, reduced_squared, _mm256_set1_ps(0x1.55553cp-5f));
    cosine = _mm256_fmadd_ps(
        cosine, reduced_squared, _mm256_set1_ps(-0x1.000000p-1f));
    cosine = _mm256_fmadd_ps(
        cosine, reduced_squared, _mm256_set1_ps(0x1.000000p+0f));

    __m256i integer_quadrant = _mm256_cvtps_epi32(quadrant);
    __m256 sine_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
        _mm256_and_si256(integer_quadrant, one), _mm256_setzero_si256()));
    __m256 result = _mm256_blendv_ps(cosine, sine, sine_mask);
    __m256 negate_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
        _mm256_and_si256(integer_quadrant, two), two));
    result = _mm256_blendv_ps(
        result, _mm256_sub_ps(zero, result), negate_mask);
    _mm256_storeu_ps(destination, result);

    if (range_bits != 0xff) {
        for (int lane = 0; lane < 8; ++lane) {
            if (!(range_bits & (1 << lane)))
                destination[lane] = sinf(source[lane]);
        }
    }
}

__declspec(noinline) void cnp_avx2_sin_f32(
    const float *source, float *destination, int64_t n) {
    int64_t index = 0;
    for (; index + 7 < n; index += 8)
        cnp_avx2_sin_f32_block(source + index, destination + index);
    if (index < n) {
        float input_tail[8] = {0.0f};
        float output_tail[8];
        int64_t remaining = n - index;
        memcpy(input_tail, source + index, (size_t)remaining * sizeof(float));
        cnp_avx2_sin_f32_block(input_tail, output_tail);
        memcpy(destination + index, output_tail,
               (size_t)remaining * sizeof(float));
    }
}

/*
 * NumPy 1.25.0 tanh FMA3 kernels, ported from
 * numpy/core/src/umath/loops_hyperbolic.dispatch.c.src.  The coefficient
 * tables and Horner order are intentionally bit-for-bit identical to the
 * NumPy AVX2/FMA3 dispatch selected on supported Windows hosts.
 *
 * Copyright (c) 2005-2023, NumPy Developers. All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the NumPy Developers nor the names of contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
static const uint32_t cnp_numpy_125_tanh_f32_lut[32][8] = {
    {0xbc0e2f66, 0x3e0910e9, 0xb76dd6b9, 0xbeaaaaa5,
     0xb0343c7b, 0x3f800000, 0x00000000, 0x00000000},
    {0x460bda12, 0x43761143, 0xbe1c276d, 0xbeab0612,
     0xbd6ee69d, 0x3f7f1f84, 0x3d6fb9c9, 0x3d700000},
    {0x43d638ef, 0x4165ecdc, 0x3c1dcf2f, 0xbea7f01f,
     0xbd8f0da7, 0x3f7ebd11, 0x3d8fc35f, 0x3d900000},
    {0xc3e11c3e, 0xc190f756, 0x3dc1a78d, 0xbea4e120,
     0xbdae477d, 0x3f7e1e5f, 0x3daf9169, 0x3db00000},
    {0xc2baa4e9, 0xc08c097d, 0x3d96f985, 0xbea387b7,
     0xbdcd2a1f, 0x3f7d609f, 0x3dcf49ab, 0x3dd00000},
    {0xc249da2d, 0xc02ba813, 0x3da2b61b, 0xbea15962,
     0xbdeba80d, 0x3f7c842d, 0x3deee849, 0x3df00000},
    {0xc1859b82, 0xbf7f6bda, 0x3dc13397, 0xbe9d57f7,
     0xbe0c443b, 0x3f7b00e5, 0x3e0f0ee8, 0x3e100000},
    {0x40dd5b57, 0x3f2b1dc0, 0x3dd2f670, 0xbe976b5a,
     0xbe293cf3, 0x3f789580, 0x3e2e4984, 0x3e300000},
    {0x40494640, 0x3ece105d, 0x3df48a0a, 0xbe90230d,
     0xbe44f282, 0x3f75b8ad, 0x3e4d2f8e, 0x3e500000},
    {0x40c730a8, 0x3f426a94, 0x3e06c5a8, 0xbe880dff,
     0xbe5f3651, 0x3f726fd9, 0x3e6bb32e, 0x3e700000},
    {0xbf0f160e, 0xbadb0dc4, 0x3e1a3aba, 0xbe7479b3,
     0xbe81c7c0, 0x3f6cc59b, 0x3e8c51cd, 0x3e900000},
    {0x3e30e76f, 0x3da43b17, 0x3e27c405, 0xbe4c3d88,
     0xbe96d7ca, 0x3f63fb92, 0x3ea96163, 0x3eb00000},
    {0xbea81387, 0xbd51ab88, 0x3e2e78d0, 0xbe212482,
     0xbea7fb8e, 0x3f59ff97, 0x3ec543f1, 0x3ed00000},
    {0xbdb26a1c, 0xbcaea23d, 0x3e2c3e44, 0xbdeb8cba,
     0xbeb50e9e, 0x3f4f11d7, 0x3edfd735, 0x3ef00000},
    {0xbd351e57, 0xbd3b6d8d, 0x3e1d3097, 0xbd5e78ad,
     0xbec12efe, 0x3f3d7573, 0x3f028438, 0x3f100000},
    {0xbb4c01a0, 0xbd6caaad, 0x3df4a8f4, 0x3c6b5e6e,
     0xbec4be92, 0x3f24f360, 0x3f18abf0, 0x3f300000},
    {0x3c1d7bfb, 0xbd795bed, 0x3da38508, 0x3d839143,
     0xbebce070, 0x3f0cbfe7, 0x3f2bc480, 0x3f500000},
    {0x3c722cd1, 0xbd5fddda, 0x3d31416a, 0x3dc21ee1,
     0xbead510e, 0x3eec1a69, 0x3f3bec1c, 0x3f700000},
    {0x3c973f1c, 0xbd038f3b, 0x3b562657, 0x3de347af,
     0xbe8ef7d6, 0x3eb0a801, 0x3f4f2e5b, 0x3f900000},
    {0x3c33a31b, 0xbc1cad63, 0xbcaeeac9, 0x3dcbec96,
     0xbe4b8704, 0x3e6753a2, 0x3f613c53, 0x3fb00000},
    {0x3b862ef4, 0x3abb4766, 0xbcce9419, 0x3d99ef2d,
     0xbe083237, 0x3e132f1a, 0x3f6ce37d, 0x3fd00000},
    {0x3a27b3d0, 0x3b95f10b, 0xbcaaeac4, 0x3d542ea1,
     0xbdaf7449, 0x3db7e7d3, 0x3f743c4f, 0x3ff00000},
    {0xba3b5907, 0x3b825873, 0xbc49e7d0, 0x3cdde701,
     0xbd2e1ec4, 0x3d320845, 0x3f7a5feb, 0x40100000},
    {0xba0efc22, 0x3afaea66, 0xbba71ddd, 0x3c2cca67,
     0xbc83bf06, 0x3c84d3d4, 0x3f7dea85, 0x40300000},
    {0xb97f9f0f, 0x3a49f878, 0xbb003b0e, 0x3b81cb27,
     0xbbc3e0b5, 0x3bc477b7, 0x3f7f3b3d, 0x40500000},
    {0xb8c8af50, 0x39996bf3, 0xba3f9a05, 0x3ac073a1,
     0xbb10aadc, 0x3b10d3da, 0x3f7fb78c, 0x40700000},
    {0xb7bdddfb, 0x388f3e6c, 0xb92c08a7, 0x39ac3032,
     0xba0157db, 0x3a01601e, 0x3f7fefd4, 0x40900000},
    {0xb64f2950, 0x371bb0e3, 0xb7ba9232, 0x383a94d9,
     0xb88c18f2, 0x388c1a3b, 0x3f7ffdd0, 0x40b00000},
    {0xb4e085b1, 0x35a8a5e6, 0xb64a0b0f, 0x36ca081d,
     0xb717b096, 0x3717b0da, 0x3f7fffb4, 0x40d00000},
    {0xb3731dfa, 0x34369b17, 0xb4dac169, 0x355abd4c,
     0xb5a43bae, 0x35a43bce, 0x3f7ffff6, 0x40f00000},
    {0xb15a1f04, 0x322487b0, 0xb2ab78ac, 0x332b3cb6,
     0xb383012c, 0x338306c6, 0x3f7fffff, 0x41100000},
    {0x00000000, 0x00000000, 0x00000000, 0x00000000,
     0x00000000, 0x00000000, 0x3f800000, 0x00000000},
};

static const uint64_t cnp_numpy_125_tanh_f64_lut[16][18] = {
    {0x0000000000000000ull, 0x0000000000000000ull,
     0x3ff0000000000000ull, 0xbbf0b3ea3fdfaa19ull,
     0xbfd5555555555555ull, 0xbce6863ee44ed636ull,
     0x3fc1111111112ab5ull, 0xbda1ea19ddddb3b4ull,
     0xbfaba1ba1990520bull, 0xbe351ca7f096011full,
     0x3f9664f94e6ac14eull, 0xbea8c4c1fd7852feull,
     0xbf822404577aa9ddull, 0xbefdd99a221ed573ull,
     0x3f6e3be689423841ull, 0xbf2a1306713a4f3aull,
     0xbf55d7e76dc56871ull, 0x3f35e67ab76a26e7ull},
    {0x3fcc000000000000ull, 0x3fcb8fd0416a7c92ull,
     0x3fee842ca3f08532ull, 0xbfca48aaeb53bc21ull,
     0xbfd183afc292ba11ull, 0x3fc04dcd0476c75eull,
     0x3fb5c19efdfc08adull, 0xbfb0b8df995ce4dfull,
     0xbf96e37bba52f6fcull, 0x3f9eaaf3320c3851ull,
     0xbf94d3343bae39ddull, 0xbfccce16b1046f13ull,
     0x403d8b07f7a82aa3ull, 0x4070593a3735bab4ull,
     0xc0d263511f5baac1ull, 0xc1045e509116b066ull,
     0x41528c38809c90c7ull, 0x41848ee0627d8206ull},
    {0x3fd4000000000000ull, 0x3fd35f98a0ea650eull,
     0x3fed11574af58f1bull, 0xbfd19921f4329916ull,
     0xbfcc1a4b039c9bfaull, 0x3fc43d3449a80f08ull,
     0x3fa74c98dc34fbacull, 0xbfb2955cf41e8164ull,
     0x3ecff7df18455399ull, 0x3f9cf823fe761fc1ull,
     0xbf7bc748e60df843ull, 0xbf81a16f224bb7b6ull,
     0xbf9f44ab92fbab0aull, 0xbfccab654e44835eull,
     0x40169f73b15ebe5cull, 0x4041fab9250984ceull,
     0xc076d57fb5190b02ull, 0xc0a216d618b489ecull},
    {0x3fdc000000000000ull, 0x3fda5729ee488037ull,
     0x3fea945b9c24e4f9ull, 0xbfd5e0f09bef8011ull,
     0xbfc16e1e6d8d0be6ull, 0x3fc5c26f3699b7e7ull,
     0xbf790d6a8eff0a77ull, 0xbfaf9d05c309f7c6ull,
     0x3f97362834d33a4eull, 0x3f9022271754ff1full,
     0xbf8c89372b43ba85ull, 0xbf62cbf00406bc09ull,
     0x3fb2eac604473d6aull, 0x3fd13ed80037dbacull,
     0xc025c1dd41cd6cb5ull, 0xc0458d090ec3de95ull,
     0x4085f09f888f8adaull, 0x40a5b89107c8af4full},
    {0x3fe4000000000000ull, 0x3fe1bf47eabb8f95ull,
     0x3fe6284c3374f815ull, 0xbfd893b59c35c882ull,
     0xbf92426c751e48a2ull, 0x3fc1a686f6ab2533ull,
     0xbfac3c021789a786ull, 0xbf987d27ccff4291ull,
     0x3f9e7f8380184b45ull, 0xbf731fe77c9c60afull,
     0xbf8129a092de747aull, 0x3f75b29bb02cf69bull,
     0x3f45f87d903aaac8ull, 0xbf6045b9076cc487ull,
     0xbf58fd89fe05e0d1ull, 0xbf74949d60113d63ull,
     0x3fa246332a2fcba5ull, 0x3fb69d8374520edaull},
    {0x3fec000000000000ull, 0x3fe686650b8c2015ull,
     0x3fe02500a09f8d6eull, 0xbfd6ba7cb7576538ull,
     0x3fb4f152b2bad124ull, 0x3faf203c316ce730ull,
     0xbfae2196b7326859ull, 0x3f8b2ca62572b098ull,
     0x3f869543e7c420d4ull, 0xbf84a6046865ec7dull,
     0x3f60c85b4d538746ull, 0x3f607df0f9f90c17ull,
     0xbf5e104671036300ull, 0x3f2085ee7e8ac170ull,
     0x3f73f7af01d5af7aull, 0x3f7c9fd6200d0adeull,
     0xbfb29d851a896fcdull, 0xbfbded519f981716ull},
    {0x3ff4000000000000ull, 0x3feb2523bb6b2deeull,
     0x3fd1f25131e3a8c0ull, 0xbfce7291743d7555ull,
     0x3fbbba40cbef72beull, 0xbf89c7a02788557cull,
     0xbf93a7a011ff8c2aull, 0x3f8f1cf6c7f5b00aull,
     0xbf7326bd4914222aull, 0xbf4ca3f1f2b9192bull,
     0x3f5be9392199ec18ull, 0xbf4b852a6e0758d5ull,
     0x3f19bc98ddf0f340ull, 0x3f23524622610430ull,
     0xbf1e40bdead17e6bull, 0x3f02cd40e0ad0a9full,
     0x3ed9065ae369b212ull, 0xbef02d288b5b3371ull},
    {0x3ffc000000000000ull, 0x3fee1fbf97e33527ull,
     0x3fbd22ca1c24a139ull, 0xbfbb6d85a01efb80ull,
     0x3fb01ba038be6a3dull, 0xbf98157e26e0d541ull,
     0x3f6e4709c7e8430eull, 0x3f60379811e43dd5ull,
     0xbf5fc15b0a9d98faull, 0x3f4c77dee0afd227ull,
     0xbf2a0c68a4489f10ull, 0xbf0078c63d1b8445ull,
     0x3f0d4304bc9246e8ull, 0xbeff12a6626911b4ull,
     0x3ee224cd6c4513e5ull, 0xbe858ab8e019f311ull,
     0xbeb8e1ba4c98a030ull, 0x3eb290981209c1a6ull},
    {0x4004000000000000ull, 0x3fef9258260a71c2ull,
     0x3f9b3afe1fba5c76ull, 0xbf9addae58c7141aull,
     0x3f916df44871efc8ull, 0xbf807b55c1c7d278ull,
     0x3f67682afa611151ull, 0xbf4793826f78537eull,
     0x3f14cffcfa69fbb6ull, 0x3f04055bce68597aull,
     0xbf00462601dc2faaull, 0x3eec12eadd55be7aull,
     0xbed13c415f7b9d41ull, 0x3eab9008bca408afull,
     0xbe24b645e68eeaa3ull, 0xbe792fa6323b7cf8ull,
     0x3e6ffd0766ad4016ull, 0xbe567e924bf5ff6eull},
    {0x400c000000000000ull, 0x3feff112c63a9077ull,
     0x3f6dd37d19b22b21ull, 0xbf6dc59376c7aa19ull,
     0x3f63c6869dfc8870ull, 0xbf53a18d5843190full,
     0x3f3ef2ee77717cbfull, 0xbf2405695e36240full,
     0x3f057e48e5b79d10ull, 0xbee2bf0cb4a71647ull,
     0x3eb7b6a219dea9f4ull, 0xbe6fa600f593181bull,
     0xbe722b8d9720cdb0ull, 0x3e634df71865f620ull,
     0xbe4abfebfb72bc83ull, 0x3e2df04d67876402ull,
     0xbe0c63c29f505f5bull, 0x3de3f7f7de6b0eb6ull},
    {0x4014000000000000ull, 0x3fefff419668df11ull,
     0x3f27ccec13a9ef96ull, 0xbf27cc5e74677410ull,
     0x3f1fb9aef915d828ull, 0xbf0fb6bbc89b1a5bull,
     0x3ef95a4482f180b7ull, 0xbee0e08de39ce756ull,
     0x3ec33b66d7d77264ull, 0xbea31eaafe73efd5ull,
     0x3e80cbcc8d4c5c8aull, 0xbe5a3c935dce3f7dull,
     0x3e322666d739bec0ull, 0xbe05bb1bcf83ca73ull,
     0x3dd51c38f8695ed3ull, 0xbd95c72be95e4d2cull,
     0xbd7fab216b9e0e49ull, 0x3d69ed18bae3ebbcull},
    {0x401c000000000000ull, 0x3feffffc832750f2ull,
     0x3ecbe6c3f33250aeull, 0xbecbe6c0e8b4cc87ull,
     0x3ec299d1e27c6e11ull, 0xbeb299c9c684a963ull,
     0x3e9dc2c27da3b603ull, 0xbe83d709ba5f714eull,
     0x3e66ac4e578b9b10ull, 0xbe46abb02c4368edull,
     0x3e2425bb231a5e29ull, 0xbe001c6d95e3ae96ull,
     0x3dd76a553d7e7918ull, 0xbdaf2ac143fb6762ull,
     0x3d8313ac38c6832bull, 0xbd55a89c30203106ull,
     0x3d2826b62056aa27ull, 0xbcf7534c4f3dfa71ull},
    {0x4024000000000000ull, 0x3feffffffdc96f35ull,
     0x3e41b4865394f75full, 0xbe41b486526b0565ull,
     0x3e379b5ddcca334cull, 0xbe279b5dd4fb3d01ull,
     0x3e12e2afd9f7433eull, 0xbdf92e3fc5ee63e0ull,
     0x3ddcc74b8d3d5c42ull, 0xbdbcc749ca8079ddull,
     0x3d9992a4beac8662ull, 0xbd74755a00ea1fd3ull,
     0x3d4de0fa59416a39ull, 0xbd23eae52a3dbf57ull,
     0x3cf7787935626685ull, 0xbccad6b3bb9eff65ull,
     0x3ca313e31762f523ull, 0xbc730b73f1eaff20ull},
    {0x402c000000000000ull, 0x3fefffffffffcf58ull,
     0x3d8853f01bda5f28ull, 0xbd8853f01bef63a4ull,
     0x3d8037f57bc62c9aull, 0xbd7037f57ae72aa6ull,
     0x3d59f320348679baull, 0xbd414cc030f2110eull,
     0x3d23c589137f92b4ull, 0xbd03c5883836b9d2ull,
     0x3ce191ba5ed3fb67ull, 0xbcbc1c6c063bb7acull,
     0x3c948716cf3681b4ull, 0xbc6b5e3e9ca0955eull,
     0x3c401ffc49c6bc29ull, 0xbc12705ccd3dd884ull,
     0x3bea37aa21895319ull, 0xbbba2cff8135d462ull},
    {0x4034000000000000ull, 0x3ff0000000000000ull,
     0x3c73953c0197ef58ull, 0xbc73955be519be31ull,
     0x3c6a2d4b50a2cff7ull, 0xbc5a2ca2bba78e86ull,
     0x3c44b61d9bbcc940ull, 0xbc2ba022e8d82a87ull,
     0x3c107f8e2c8707a1ull, 0xbbf07a5416264aecull,
     0x3bc892450bad44c4ull, 0xbba3be9a4460fe00ull,
     0x3b873f9f2d2fda99ull, 0xbb5eca68e2c1ba2eull,
     0xbabf0b21acfa52abull, 0xba8e0a4c47ae75f5ull,
     0x3ae5c7f1fd871496ull, 0xbab5a71b5f7d9035ull},
    {0x0000000000000000ull, 0x3ff0000000000000ull,
     0x0000000000000000ull, 0x0000000000000000ull,
     0x0000000000000000ull, 0x0000000000000000ull,
     0x0000000000000000ull, 0x0000000000000000ull,
     0x0000000000000000ull, 0x0000000000000000ull,
     0x0000000000000000ull, 0x0000000000000000ull,
     0x0000000000000000ull, 0x0000000000000000ull,
     0x0000000000000000ull, 0x0000000000000000ull,
     0x0000000000000000ull, 0x0000000000000000ull},
};

static __forceinline __m256 cnp_avx2_gather_tanh_f32(
    __m256i row_offsets, int coefficient) {
    __m256i indices = _mm256_add_epi32(
        row_offsets, _mm256_set1_epi32(coefficient));
    return _mm256_i32gather_ps(
        (const float*)cnp_numpy_125_tanh_f32_lut, indices, 4);
}

static __forceinline __m256d cnp_avx2_gather_tanh_f64(
    __m256i row_offsets, int coefficient) {
    __m256i indices = _mm256_add_epi64(
        row_offsets, _mm256_set1_epi64x(coefficient));
    return _mm256_i64gather_pd(
        (const double*)cnp_numpy_125_tanh_f64_lut, indices, 8);
}

static __forceinline void cnp_avx2_tanh_f32_block(
    const float *source, float *destination) {
    const __m256i magnitude_mask = _mm256_set1_epi32(0x7fffffff);
    const __m256i sign_mask = _mm256_set1_epi32((int)0x80000000u);
    __m256 input = _mm256_loadu_ps(source);
    __m256i input_bits = _mm256_castps_si256(input);
    __m256i ndnan = _mm256_and_si256(
        input_bits, _mm256_set1_epi32(0x7fe00000));
    __m256i special = _mm256_cmpgt_epi32(
        _mm256_set1_epi32(0x7f000001), ndnan);
    __m256 not_nan = _mm256_cmp_ps(input, input, _CMP_ORD_Q);
    __m256i indices = _mm256_sub_epi32(
        ndnan, _mm256_set1_epi32(0x3d400000));
    indices = _mm256_max_epi32(indices, _mm256_setzero_si256());
    indices = _mm256_min_epi32(indices, _mm256_set1_epi32(0x03e00000));
    indices = _mm256_srli_epi32(indices, 21);
    __m256i row_offsets = _mm256_slli_epi32(indices, 3);

    __m256 c6 = cnp_avx2_gather_tanh_f32(row_offsets, 0);
    __m256 c5 = cnp_avx2_gather_tanh_f32(row_offsets, 1);
    __m256 c4 = cnp_avx2_gather_tanh_f32(row_offsets, 2);
    __m256 c3 = cnp_avx2_gather_tanh_f32(row_offsets, 3);
    __m256 c2 = cnp_avx2_gather_tanh_f32(row_offsets, 4);
    __m256 c1 = cnp_avx2_gather_tanh_f32(row_offsets, 5);
    __m256 c0 = cnp_avx2_gather_tanh_f32(row_offsets, 6);
    __m256 b = cnp_avx2_gather_tanh_f32(row_offsets, 7);
    __m256 magnitude = _mm256_castsi256_ps(
        _mm256_and_si256(input_bits, magnitude_mask));
    __m256 y = _mm256_sub_ps(magnitude, b);
    __m256 result = _mm256_fmadd_ps(c6, y, c5);
    result = _mm256_fmadd_ps(result, y, c4);
    result = _mm256_fmadd_ps(result, y, c3);
    result = _mm256_fmadd_ps(result, y, c2);
    result = _mm256_fmadd_ps(result, y, c1);
    result = _mm256_fmadd_ps(result, y, c0);
    result = _mm256_blendv_ps(
        _mm256_set1_ps(1.0f), result, _mm256_castsi256_ps(special));
    result = _mm256_castsi256_ps(_mm256_or_si256(
        _mm256_castps_si256(result),
        _mm256_and_si256(input_bits, sign_mask)));
    result = _mm256_blendv_ps(
        _mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000)),
        result, not_nan);
    _mm256_storeu_ps(destination, result);
}

static __forceinline void cnp_avx2_tanh_f64_block(
    const double *source, double *destination) {
    const __m256i magnitude_mask = _mm256_set1_epi64x(
        0x7fffffffffffffffull);
    const __m256i sign_mask = _mm256_set1_epi64x(
        (int64_t)0x8000000000000000ull);
    __m256d input = _mm256_loadu_pd(source);
    __m256i input_bits = _mm256_castpd_si256(input);
    __m256i ndnan = _mm256_and_si256(
        input_bits, _mm256_set1_epi64x(0x7ff8000000000000ull));
    __m256i special = _mm256_cmpgt_epi64(
        _mm256_set1_epi64x(0x7fe0000000000001ull), ndnan);
    __m256d not_nan = _mm256_cmp_pd(input, input, _CMP_ORD_Q);
    __m256i indices = _mm256_sub_epi64(
        ndnan, _mm256_set1_epi64x(0x3fc0000000000000ull));
    __m256i below = _mm256_cmpgt_epi64(_mm256_setzero_si256(), indices);
    indices = _mm256_blendv_epi8(indices, _mm256_setzero_si256(), below);
    __m256i maximum = _mm256_set1_epi64x(0x0078000000000000ull);
    __m256i above = _mm256_cmpgt_epi64(indices, maximum);
    indices = _mm256_blendv_epi8(indices, maximum, above);
    indices = _mm256_srli_epi64(indices, 51);
    __m256i row_offsets = _mm256_add_epi64(
        _mm256_slli_epi64(indices, 4), _mm256_slli_epi64(indices, 1));

    __m256d b = cnp_avx2_gather_tanh_f64(row_offsets, 0);
    __m256d c0 = cnp_avx2_gather_tanh_f64(row_offsets, 1);
    __m256d c1 = cnp_avx2_gather_tanh_f64(row_offsets, 2);
    __m256d c2 = cnp_avx2_gather_tanh_f64(row_offsets, 3);
    __m256d c3 = cnp_avx2_gather_tanh_f64(row_offsets, 4);
    __m256d c4 = cnp_avx2_gather_tanh_f64(row_offsets, 5);
    __m256d c5 = cnp_avx2_gather_tanh_f64(row_offsets, 6);
    __m256d c6 = cnp_avx2_gather_tanh_f64(row_offsets, 7);
    __m256d c7 = cnp_avx2_gather_tanh_f64(row_offsets, 8);
    __m256d c8 = cnp_avx2_gather_tanh_f64(row_offsets, 9);
    __m256d c9 = cnp_avx2_gather_tanh_f64(row_offsets, 10);
    __m256d c10 = cnp_avx2_gather_tanh_f64(row_offsets, 11);
    __m256d c11 = cnp_avx2_gather_tanh_f64(row_offsets, 12);
    __m256d c12 = cnp_avx2_gather_tanh_f64(row_offsets, 13);
    __m256d c13 = cnp_avx2_gather_tanh_f64(row_offsets, 14);
    __m256d c14 = cnp_avx2_gather_tanh_f64(row_offsets, 15);
    __m256d c15 = cnp_avx2_gather_tanh_f64(row_offsets, 16);
    __m256d c16 = cnp_avx2_gather_tanh_f64(row_offsets, 17);
    __m256d magnitude = _mm256_castsi256_pd(
        _mm256_and_si256(input_bits, magnitude_mask));
    __m256d y = _mm256_sub_pd(magnitude, b);
    __m256d result = _mm256_fmadd_pd(c16, y, c15);
    result = _mm256_fmadd_pd(result, y, c14);
    result = _mm256_fmadd_pd(result, y, c13);
    result = _mm256_fmadd_pd(result, y, c12);
    result = _mm256_fmadd_pd(result, y, c11);
    result = _mm256_fmadd_pd(result, y, c10);
    result = _mm256_fmadd_pd(result, y, c9);
    result = _mm256_fmadd_pd(result, y, c8);
    result = _mm256_fmadd_pd(result, y, c7);
    result = _mm256_fmadd_pd(result, y, c6);
    result = _mm256_fmadd_pd(result, y, c5);
    result = _mm256_fmadd_pd(result, y, c4);
    result = _mm256_fmadd_pd(result, y, c3);
    result = _mm256_fmadd_pd(result, y, c2);
    result = _mm256_fmadd_pd(result, y, c1);
    result = _mm256_fmadd_pd(result, y, c0);
    result = _mm256_blendv_pd(
        _mm256_set1_pd(1.0), result, _mm256_castsi256_pd(special));
    result = _mm256_castsi256_pd(_mm256_or_si256(
        _mm256_castpd_si256(result),
        _mm256_and_si256(input_bits, sign_mask)));
    result = _mm256_blendv_pd(
        _mm256_castsi256_pd(_mm256_set1_epi64x(0x7ff8000000000000ull)),
        result, not_nan);
    _mm256_storeu_pd(destination, result);
}

__declspec(noinline) void cnp_avx2_tanh_f32(
    const float *source, float *destination, int64_t n) {
    int64_t index = 0;
    for (; index + 7 < n; index += 8)
        cnp_avx2_tanh_f32_block(source + index, destination + index);
    if (index < n) {
        float input_tail[8] = {0.0f};
        float output_tail[8];
        int64_t remaining = n - index;
        memcpy(input_tail, source + index, (size_t)remaining * sizeof(float));
        cnp_avx2_tanh_f32_block(input_tail, output_tail);
        memcpy(destination + index, output_tail,
               (size_t)remaining * sizeof(float));
    }
}

__declspec(noinline) void cnp_avx2_tanh_f64(
    const double *source, double *destination, int64_t n) {
    int64_t index = 0;
    for (; index + 3 < n; index += 4)
        cnp_avx2_tanh_f64_block(source + index, destination + index);
    if (index < n) {
        double input_tail[4] = {0.0};
        double output_tail[4];
        int64_t remaining = n - index;
        memcpy(input_tail, source + index, (size_t)remaining * sizeof(double));
        cnp_avx2_tanh_f64_block(input_tail, output_tail);
        memcpy(destination + index, output_tail,
               (size_t)remaining * sizeof(double));
    }
}

__declspec(noinline) void cnp_avx2_arange(
    double *out, double start, double step, int64_t n) {
    const __m256d lane03 = _mm256_set_pd(
        3.0 * step, 2.0 * step, step, 0.0);
    const __m256d lane47 = _mm256_set_pd(
        7.0 * step, 6.0 * step, 5.0 * step, 4.0 * step);
    const __m256d lane811 = _mm256_set_pd(
        11.0 * step, 10.0 * step, 9.0 * step, 8.0 * step);
    const __m256d lane1215 = _mm256_set_pd(
        15.0 * step, 14.0 * step, 13.0 * step, 12.0 * step);
    int64_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m256d base = _mm256_set1_pd(start + (double)i * step);
        _mm256_storeu_pd(out + i, _mm256_add_pd(base, lane03));
        _mm256_storeu_pd(out + i + 4, _mm256_add_pd(base, lane47));
        _mm256_storeu_pd(out + i + 8, _mm256_add_pd(base, lane811));
        _mm256_storeu_pd(out + i + 12, _mm256_add_pd(base, lane1215));
    }
    for (; i < n; ++i) out[i] = start + (double)i * step;
}

#define CNP_AVX2_BINARY_BODY(vector_operation, scalar_operator) \
    int64_t i = 0; \
    for (; i + 15 < n; i += 16) { \
        _mm256_storeu_pd(out + i, vector_operation( \
            _mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i))); \
        _mm256_storeu_pd(out + i + 4, vector_operation( \
            _mm256_loadu_pd(a + i + 4), _mm256_loadu_pd(b + i + 4))); \
        _mm256_storeu_pd(out + i + 8, vector_operation( \
            _mm256_loadu_pd(a + i + 8), _mm256_loadu_pd(b + i + 8))); \
        _mm256_storeu_pd(out + i + 12, vector_operation( \
            _mm256_loadu_pd(a + i + 12), _mm256_loadu_pd(b + i + 12))); \
    } \
    for (; i + 3 < n; i += 4) { \
        _mm256_storeu_pd(out + i, vector_operation( \
            _mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i))); \
    } \
    for (; i < n; ++i) out[i] = a[i] scalar_operator b[i]

__declspec(noinline) void cnp_avx2_add(
    const double *a, const double *b, double *out, int64_t n) {
    CNP_AVX2_BINARY_BODY(_mm256_add_pd, +);
}

__declspec(noinline) void cnp_avx2_subtract(
    const double *a, const double *b, double *out, int64_t n) {
    CNP_AVX2_BINARY_BODY(_mm256_sub_pd, -);
}

__declspec(noinline) void cnp_avx2_multiply(
    const double *a, const double *b, double *out, int64_t n) {
    CNP_AVX2_BINARY_BODY(_mm256_mul_pd, *);
}

__declspec(noinline) void cnp_avx2_divide(
    const double *a, const double *b, double *out, int64_t n) {
    CNP_AVX2_BINARY_BODY(_mm256_div_pd, /);
}

#undef CNP_AVX2_BINARY_BODY

static bool cnp_avx2_scalar_is_nan(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) ==
               UINT64_C(0x7ff0000000000000) &&
           (bits & UINT64_C(0x000fffffffffffff)) != 0;
}

static __forceinline __m256d cnp_avx2_maximum_vector(
    __m256d left, __m256d right) {
    __m256d left_nan = _mm256_cmp_pd(left, left, _CMP_UNORD_Q);
    __m256d right_nan = _mm256_cmp_pd(right, right, _CMP_UNORD_Q);
    __m256d unordered = _mm256_or_pd(left_nan, right_nan);
    __m256d nan_value = _mm256_blendv_pd(right, left, left_nan);
    return _mm256_blendv_pd(
        _mm256_max_pd(left, right), nan_value, unordered);
}

static __forceinline __m256d cnp_avx2_minimum_vector(
    __m256d left, __m256d right) {
    __m256d left_nan = _mm256_cmp_pd(left, left, _CMP_UNORD_Q);
    __m256d right_nan = _mm256_cmp_pd(right, right, _CMP_UNORD_Q);
    __m256d unordered = _mm256_or_pd(left_nan, right_nan);
    __m256d nan_value = _mm256_blendv_pd(right, left, left_nan);
    return _mm256_blendv_pd(
        _mm256_min_pd(left, right), nan_value, unordered);
}

static __forceinline __m256d cnp_avx2_fmax_vector(
    __m256d left, __m256d right) {
    __m256d right_nan = _mm256_cmp_pd(right, right, _CMP_UNORD_Q);
    __m256d result = _mm256_blendv_pd(
        _mm256_max_pd(left, right), left, right_nan);
    __m256d zero = _mm256_setzero_pd();
    __m256d both_zero = _mm256_and_pd(
        _mm256_cmp_pd(left, zero, _CMP_EQ_OQ),
        _mm256_cmp_pd(right, zero, _CMP_EQ_OQ));
    return _mm256_blendv_pd(
        result, _mm256_and_pd(left, right), both_zero);
}

static __forceinline __m256d cnp_avx2_fmin_vector(
    __m256d left, __m256d right) {
    __m256d right_nan = _mm256_cmp_pd(right, right, _CMP_UNORD_Q);
    __m256d result = _mm256_blendv_pd(
        _mm256_min_pd(left, right), left, right_nan);
    __m256d zero = _mm256_setzero_pd();
    __m256d both_zero = _mm256_and_pd(
        _mm256_cmp_pd(left, zero, _CMP_EQ_OQ),
        _mm256_cmp_pd(right, zero, _CMP_EQ_OQ));
    return _mm256_blendv_pd(
        result, _mm256_or_pd(left, right), both_zero);
}

#define CNP_AVX2_EXTREMA_MAXIMUM 0
#define CNP_AVX2_EXTREMA_MINIMUM 1
#define CNP_AVX2_EXTREMA_FMAX 2
#define CNP_AVX2_EXTREMA_FMIN 3

static void cnp_avx2_scalar_extrema(
    double left, double right, double *result, int operation) {
    uint64_t left_bits;
    uint64_t right_bits;
    memcpy(&left_bits, &left, sizeof(left_bits));
    memcpy(&right_bits, &right, sizeof(right_bits));
    bool left_nan = cnp_avx2_scalar_is_nan(left);
    bool right_nan = cnp_avx2_scalar_is_nan(right);
    uint64_t selected_bits;
    if (operation == CNP_AVX2_EXTREMA_FMAX ||
            operation == CNP_AVX2_EXTREMA_FMIN) {
        if (left_nan) selected_bits = right_nan ? left_bits : right_bits;
        else if (right_nan) selected_bits = left_bits;
        else if (left == 0.0 && right == 0.0) {
            selected_bits = operation == CNP_AVX2_EXTREMA_FMAX
                ? left_bits & right_bits : left_bits | right_bits;
        } else {
            bool select_left = operation == CNP_AVX2_EXTREMA_FMAX
                ? left > right : left < right;
            selected_bits = select_left ? left_bits : right_bits;
        }
    } else if (left_nan) selected_bits = left_bits;
    else if (right_nan) selected_bits = right_bits;
    else {
        bool select_left = operation == CNP_AVX2_EXTREMA_MAXIMUM
            ? left > right : left < right;
        selected_bits = select_left ? left_bits : right_bits;
    }
    memcpy(result, &selected_bits, sizeof(selected_bits));
}

#define CNP_AVX2_EXTREMA_BODY(vector_function, scalar_operation) \
    int64_t i = 0; \
    for (; i + 15 < n; i += 16) { \
        _mm256_storeu_pd(out + i, vector_function( \
            _mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i))); \
        _mm256_storeu_pd(out + i + 4, vector_function( \
            _mm256_loadu_pd(a + i + 4), _mm256_loadu_pd(b + i + 4))); \
        _mm256_storeu_pd(out + i + 8, vector_function( \
            _mm256_loadu_pd(a + i + 8), _mm256_loadu_pd(b + i + 8))); \
        _mm256_storeu_pd(out + i + 12, vector_function( \
            _mm256_loadu_pd(a + i + 12), _mm256_loadu_pd(b + i + 12))); \
    } \
    for (; i + 3 < n; i += 4) { \
        _mm256_storeu_pd(out + i, vector_function( \
            _mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i))); \
    } \
    for (; i < n; ++i) \
        cnp_avx2_scalar_extrema(a[i], b[i], out + i, scalar_operation)

__declspec(noinline) void cnp_avx2_maximum(
    const double *a, const double *b, double *out, int64_t n) {
    CNP_AVX2_EXTREMA_BODY(
        cnp_avx2_maximum_vector, CNP_AVX2_EXTREMA_MAXIMUM);
}

__declspec(noinline) void cnp_avx2_minimum(
    const double *a, const double *b, double *out, int64_t n) {
    CNP_AVX2_EXTREMA_BODY(
        cnp_avx2_minimum_vector, CNP_AVX2_EXTREMA_MINIMUM);
}

__declspec(noinline) void cnp_avx2_fmax(
    const double *a, const double *b, double *out, int64_t n) {
    CNP_AVX2_EXTREMA_BODY(
        cnp_avx2_fmax_vector, CNP_AVX2_EXTREMA_FMAX);
}

__declspec(noinline) void cnp_avx2_fmin(
    const double *a, const double *b, double *out, int64_t n) {
    CNP_AVX2_EXTREMA_BODY(
        cnp_avx2_fmin_vector, CNP_AVX2_EXTREMA_FMIN);
}

#undef CNP_AVX2_EXTREMA_BODY
#undef CNP_AVX2_EXTREMA_MAXIMUM
#undef CNP_AVX2_EXTREMA_MINIMUM
#undef CNP_AVX2_EXTREMA_FMAX
#undef CNP_AVX2_EXTREMA_FMIN

#define CNP_AVX2_LOGICAL_AND 0
#define CNP_AVX2_LOGICAL_OR 1
#define CNP_AVX2_LOGICAL_XOR 2

static __forceinline bool cnp_avx2_scalar_truth_f64(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7fffffffffffffff)) != 0;
}

static __forceinline int cnp_avx2_truth_mask_f64x4(
    const double *values) {
    __m256i magnitude = _mm256_and_si256(
        _mm256_loadu_si256((const __m256i*)values),
        _mm256_set1_epi64x(INT64_MAX));
    __m256i zero64 = _mm256_cmpeq_epi64(
        magnitude, _mm256_setzero_si256());
    int zero_mask = _mm256_movemask_pd(
        _mm256_castsi256_pd(zero64));
    return zero_mask ^ 15;
}

static __forceinline uint32_t cnp_avx2_pack_bool4(int mask) {
    return (uint32_t)(
        (mask & 1) |
        ((mask & 2) << 7) |
        ((mask & 4) << 14) |
        ((mask & 8) << 21));
}

static __forceinline void cnp_avx2_logical_binary4(
    const double *a,
    const double *b,
    uint8_t *out,
    int operation) {
    int left_mask = cnp_avx2_truth_mask_f64x4(a);
    int right_mask = cnp_avx2_truth_mask_f64x4(b);
    int result_mask = operation == CNP_AVX2_LOGICAL_AND
        ? left_mask & right_mask
        : operation == CNP_AVX2_LOGICAL_OR
        ? left_mask | right_mask
        : left_mask ^ right_mask;
    uint32_t packed = cnp_avx2_pack_bool4(result_mask);
    memcpy(out, &packed, sizeof(packed));
}

static void cnp_avx2_logical_binary(
    const double *a,
    const double *b,
    uint8_t *out,
    int64_t n,
    int operation) {
    int64_t i = 0;
    for (; i + 15 < n; i += 16) {
        cnp_avx2_logical_binary4(a + i, b + i, out + i, operation);
        cnp_avx2_logical_binary4(
            a + i + 4, b + i + 4, out + i + 4, operation);
        cnp_avx2_logical_binary4(
            a + i + 8, b + i + 8, out + i + 8, operation);
        cnp_avx2_logical_binary4(
            a + i + 12, b + i + 12, out + i + 12, operation);
    }
    for (; i + 3 < n; i += 4)
        cnp_avx2_logical_binary4(a + i, b + i, out + i, operation);
    for (; i < n; ++i) {
        bool left = cnp_avx2_scalar_truth_f64(a[i]);
        bool right = cnp_avx2_scalar_truth_f64(b[i]);
        out[i] = (uint8_t)(operation == CNP_AVX2_LOGICAL_AND
            ? left && right
            : operation == CNP_AVX2_LOGICAL_OR
            ? left || right
            : left != right);
    }
}

__declspec(noinline) void cnp_avx2_logical_and(
    const double *a, const double *b, uint8_t *out, int64_t n) {
    cnp_avx2_logical_binary(
        a, b, out, n, CNP_AVX2_LOGICAL_AND);
}

__declspec(noinline) void cnp_avx2_logical_or(
    const double *a, const double *b, uint8_t *out, int64_t n) {
    cnp_avx2_logical_binary(
        a, b, out, n, CNP_AVX2_LOGICAL_OR);
}

__declspec(noinline) void cnp_avx2_logical_xor(
    const double *a, const double *b, uint8_t *out, int64_t n) {
    cnp_avx2_logical_binary(
        a, b, out, n, CNP_AVX2_LOGICAL_XOR);
}

__declspec(noinline) void cnp_avx2_logical_not(
    const double *a, uint8_t *out, int64_t n) {
    int64_t i = 0;
    for (; i + 15 < n; i += 16) {
        for (int block = 0; block < 4; ++block) {
            int truth_mask = cnp_avx2_truth_mask_f64x4(
                a + i + block * 4);
            uint32_t packed = cnp_avx2_pack_bool4(truth_mask ^ 15);
            memcpy(out + i + block * 4, &packed, sizeof(packed));
        }
    }
    for (; i + 3 < n; i += 4) {
        int truth_mask = cnp_avx2_truth_mask_f64x4(a + i);
        uint32_t packed = cnp_avx2_pack_bool4(truth_mask ^ 15);
        memcpy(out + i, &packed, sizeof(packed));
    }
    for (; i < n; ++i)
        out[i] = (uint8_t)!cnp_avx2_scalar_truth_f64(a[i]);
}

#undef CNP_AVX2_LOGICAL_AND
#undef CNP_AVX2_LOGICAL_OR
#undef CNP_AVX2_LOGICAL_XOR

__declspec(noinline) void cnp_avx2_negative(
    const double *a, double *out, int64_t n) {
    const __m256d sign = _mm256_castsi256_pd(
        _mm256_set1_epi64x(INT64_MIN));
    int64_t i = 0;
    for (; i + 15 < n; i += 16) {
        _mm256_storeu_pd(out + i,
            _mm256_xor_pd(_mm256_loadu_pd(a + i), sign));
        _mm256_storeu_pd(out + i + 4,
            _mm256_xor_pd(_mm256_loadu_pd(a + i + 4), sign));
        _mm256_storeu_pd(out + i + 8,
            _mm256_xor_pd(_mm256_loadu_pd(a + i + 8), sign));
        _mm256_storeu_pd(out + i + 12,
            _mm256_xor_pd(_mm256_loadu_pd(a + i + 12), sign));
    }
    for (; i + 3 < n; i += 4)
        _mm256_storeu_pd(out + i,
            _mm256_xor_pd(_mm256_loadu_pd(a + i), sign));
    for (; i < n; ++i) out[i] = -a[i];
}

__declspec(noinline) void cnp_avx2_absolute(
    const double *a, double *out, int64_t n) {
    const __m256d magnitude = _mm256_castsi256_pd(
        _mm256_set1_epi64x(INT64_MAX));
    int64_t i = 0;
    for (; i + 15 < n; i += 16) {
        _mm256_storeu_pd(out + i,
            _mm256_and_pd(_mm256_loadu_pd(a + i), magnitude));
        _mm256_storeu_pd(out + i + 4,
            _mm256_and_pd(_mm256_loadu_pd(a + i + 4), magnitude));
        _mm256_storeu_pd(out + i + 8,
            _mm256_and_pd(_mm256_loadu_pd(a + i + 8), magnitude));
        _mm256_storeu_pd(out + i + 12,
            _mm256_and_pd(_mm256_loadu_pd(a + i + 12), magnitude));
    }
    for (; i + 3 < n; i += 4)
        _mm256_storeu_pd(out + i,
            _mm256_and_pd(_mm256_loadu_pd(a + i), magnitude));
    for (; i < n; ++i) out[i] = fabs(a[i]);
}

__declspec(noinline) void cnp_avx2_sqrt(
    const double *a, double *out, int64_t n) {
    int64_t i = 0;
    for (; i + 15 < n; i += 16) {
        _mm256_storeu_pd(out + i, _mm256_sqrt_pd(_mm256_loadu_pd(a + i)));
        _mm256_storeu_pd(out + i + 4,
            _mm256_sqrt_pd(_mm256_loadu_pd(a + i + 4)));
        _mm256_storeu_pd(out + i + 8,
            _mm256_sqrt_pd(_mm256_loadu_pd(a + i + 8)));
        _mm256_storeu_pd(out + i + 12,
            _mm256_sqrt_pd(_mm256_loadu_pd(a + i + 12)));
    }
    for (; i + 3 < n; i += 4)
        _mm256_storeu_pd(out + i, _mm256_sqrt_pd(_mm256_loadu_pd(a + i)));
    for (; i < n; ++i) out[i] = sqrt(a[i]);
}

__declspec(noinline) void cnp_avx2_floor(
    const double *a, double *out, int64_t n) {
    int64_t i = 0;
    for (; i + 15 < n; i += 16) {
        _mm256_storeu_pd(out + i, _mm256_floor_pd(_mm256_loadu_pd(a + i)));
        _mm256_storeu_pd(out + i + 4,
            _mm256_floor_pd(_mm256_loadu_pd(a + i + 4)));
        _mm256_storeu_pd(out + i + 8,
            _mm256_floor_pd(_mm256_loadu_pd(a + i + 8)));
        _mm256_storeu_pd(out + i + 12,
            _mm256_floor_pd(_mm256_loadu_pd(a + i + 12)));
    }
    for (; i + 3 < n; i += 4)
        _mm256_storeu_pd(out + i, _mm256_floor_pd(_mm256_loadu_pd(a + i)));
    for (; i < n; ++i) out[i] = floor(a[i]);
}

__declspec(noinline) double cnp_avx2_dot(
    const double *a, const double *b, int64_t n) {
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = sum0;
    __m256d sum2 = sum0;
    __m256d sum3 = sum0;
    int64_t i = 0;

    for (; i + 15 < n; i += 16) {
        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(
            _mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
        sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(
            _mm256_loadu_pd(a + i + 4), _mm256_loadu_pd(b + i + 4)));
        sum2 = _mm256_add_pd(sum2, _mm256_mul_pd(
            _mm256_loadu_pd(a + i + 8), _mm256_loadu_pd(b + i + 8)));
        sum3 = _mm256_add_pd(sum3, _mm256_mul_pd(
            _mm256_loadu_pd(a + i + 12), _mm256_loadu_pd(b + i + 12)));
    }
    sum0 = _mm256_add_pd(_mm256_add_pd(sum0, sum1),
                         _mm256_add_pd(sum2, sum3));
    double lanes[4];
    _mm256_storeu_pd(lanes, sum0);
    double result = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
}
