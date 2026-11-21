/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef QMX_KERNELS_H
#define QMX_KERNELS_H

#include "qmx_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(QMX_NO_SIMD) && !defined(QMX_CUSTOM_BACKEND_SET)
    #define QMX_ENABLE_BACKEND_P54C     1
    #define QMX_ENABLE_BACKEND_MCU32    1
    #define QMX_ENABLE_BACKEND_CORTEX_M 1

    #if (defined(__i386__) || defined(__x86_64__)) && (defined(__GNUC__) || defined(__clang__))
        #if !defined(QMX_DISABLE_MMX) && (defined(__MMX__) || !defined(__DJGPP__))
            #define QMX_ENABLE_BACKEND_MMX 1
        #endif
        #if (defined(__3dNOW__) || defined(__3dnow__)) && !defined(QMX_DISABLE_3DNOW)
            #define QMX_ENABLE_BACKEND_3DNOW 1
        #endif
        #if !defined(__DJGPP__)
            #define QMX_ENABLE_BACKEND_SSE2    1
            #define QMX_ENABLE_BACKEND_AVX2    1
            #define QMX_ENABLE_BACKEND_X86_EXT 1
        #endif
    #elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        #if defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
            #define QMX_ENABLE_BACKEND_SSE2 1
        #endif
        #if defined(__AVX2__)
            #define QMX_ENABLE_BACKEND_AVX2 1
        #endif
    #endif

    #if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
        #define QMX_ENABLE_BACKEND_NEON 1
        #if (defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))) || defined(__ARM_FEATURE_DOTPROD)
            #define QMX_ENABLE_BACKEND_NEON_DOT 1
        #endif
    #endif

    #if defined(__ARM_FEATURE_DSP) && (defined(__GNUC__) || defined(__clang__))
        #define QMX_ENABLE_BACKEND_ARM_DSP 1
    #endif
#endif

struct QmxBackendOps {
    const char *name;
    void (*prepare_act)(const float * QMX_RESTRICT x,
                        int8_t      * QMX_RESTRICT xq,
                        int16_t     * QMX_RESTRICT xq16,
                        float       * QMX_RESTRICT x_scales,
                        uint32_t cols);
    void (*matmul_tiled)(float         * QMX_RESTRICT out,
                         const int8_t  * QMX_RESTRICT xq,
                         const int16_t * QMX_RESTRICT xq16,
                         const float   * QMX_RESTRICT x_scales,
                         const QmxTensor *w,
                         int accumulate);
    void (*rmsnorm)(float       * QMX_RESTRICT out,
                    const float * QMX_RESTRICT x,
                    const float * QMX_RESTRICT weight,
                    uint32_t size,
                    float eps);
    float (*dot_f32)(const float * QMX_RESTRICT a,
                     const float * QMX_RESTRICT b,
                     uint32_t n);
    void (*saxpy_f32)(float       * QMX_RESTRICT y,
                      float a,
                      const float * QMX_RESTRICT x,
                      uint32_t n);
    int32_t (*dot_q8)(const int8_t  * QMX_RESTRICT q8,
                      const int16_t * QMX_RESTRICT q16,
                      const int8_t  * QMX_RESTRICT k8,
                      uint32_t n);
    void (*rmsnorm_quant)(const float * QMX_RESTRICT x,
                          const float * QMX_RESTRICT weight,
                          int8_t      * QMX_RESTRICT xq,
                          int16_t     * QMX_RESTRICT xq16,
                          float       * QMX_RESTRICT x_scales,
                          uint32_t size,
                          float eps);
    void (*swiglu_quant)(const float * QMX_RESTRICT gate,
                         const float * QMX_RESTRICT up,
                         int8_t      * QMX_RESTRICT xq,
                         int16_t     * QMX_RESTRICT xq16,
                         float       * QMX_RESTRICT x_scales,
                         uint32_t size);
    void (*f16_to_f32)(float * QMX_RESTRICT dst,
                       const uint16_t * QMX_RESTRICT src,
                       uint32_t n);
    void (*f32_to_f16)(uint16_t * QMX_RESTRICT dst,
                       const float * QMX_RESTRICT src,
                       uint32_t n);
};

extern const int8_t   qmx_lut_ternary_s8[256][4];
extern const int8_t   qmx_lut_binary_s8[16][4];
extern const uint32_t qmx_lut_ternary_u32[256][2];
extern const uint32_t qmx_lut_binary_u32[16][2];
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
extern const int16_t  qmx_lut_ternary_i16[256][4];
extern const int16_t  qmx_lut_binary_i16_4[16][4];
extern const int16_t  qmx_lut_binary_i16_8[256][8];
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
extern const int8_t   qmx_lut_binary_s8_8[256][8];
#endif

extern const QmxBackendOps qmx_ops_scalar;
extern const QmxBackendOps qmx_ops_p54c;
extern const QmxBackendOps qmx_ops_mmx;
extern const QmxBackendOps qmx_ops_3dnow;
extern const QmxBackendOps qmx_ops_sse;
extern const QmxBackendOps qmx_ops_sse2;
extern const QmxBackendOps qmx_ops_sse3;
extern const QmxBackendOps qmx_ops_ssse3;
extern const QmxBackendOps qmx_ops_sse41;
extern const QmxBackendOps qmx_ops_sse42;
extern const QmxBackendOps qmx_ops_avx;
extern const QmxBackendOps qmx_ops_avx2;
extern const QmxBackendOps qmx_ops_avx512;
extern const QmxBackendOps qmx_ops_neon;
extern const QmxBackendOps qmx_ops_neon_dot;
extern const QmxBackendOps qmx_ops_arm_dsp;
extern const QmxBackendOps qmx_ops_mcu32;
extern const QmxBackendOps qmx_ops_cortex_m0;
extern const QmxBackendOps qmx_ops_cortex_m23;
extern const QmxBackendOps qmx_ops_cortex_m3;
extern const QmxBackendOps qmx_ops_cortex_m4;
extern const QmxBackendOps qmx_ops_cortex_m33;
extern const QmxBackendOps qmx_ops_cortex_m55;

int qmx_cpu_has_mmx(void);
int qmx_cpu_has_3dnow(void);
int qmx_cpu_has_sse(void);
int qmx_cpu_has_sse2(void);
int qmx_cpu_has_sse3(void);
int qmx_cpu_has_ssse3(void);
int qmx_cpu_has_sse41(void);
int qmx_cpu_has_sse42(void);
int qmx_cpu_has_avx(void);
int qmx_cpu_has_avx2(void);
int qmx_cpu_has_avx512(void);
int qmx_cpu_has_neon(void);
int qmx_cpu_has_neon_dot(void);
int qmx_cpu_has_arm_dsp(void);

QmxBackend qmx_detect_backend(void);
int qmx_backend_needs_xq16(QmxBackend backend);
const char *qmx_backend_name(QmxBackend backend);

void qmx_register_backend(QmxBackend backend, const QmxBackendOps *ops);
const QmxBackendOps *qmx_get_backend_ops(QmxBackend backend);
void qmx_bind_backend_ops(QmxModel *model, QmxBackend requested);

#ifdef __cplusplus
}
#endif

#endif
