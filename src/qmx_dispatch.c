#include "../include/qmx_kernels.h"

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#include <intrin.h>
#endif

#if defined(__linux__) && (defined(__arm__) || defined(__aarch64__))
#include <fcntl.h>
#include <unistd.h>
#endif

static void scalar_prepare_act(const float * QMX_RESTRICT x,
                               int8_t      * QMX_RESTRICT xq,
                               int16_t     * QMX_RESTRICT xq16,
                               float       * QMX_RESTRICT x_scales,
                               uint32_t cols) {
    (void)xq16;
    qmx_quantize_activations_q8(x, xq, x_scales, cols);
}

static void scalar_matmul_tiled(float         * QMX_RESTRICT out,
                                const int8_t  * QMX_RESTRICT xq,
                                const int16_t * QMX_RESTRICT xq16,
                                const float   * QMX_RESTRICT x_scales,
                                const QmxTensor *w,
                                int accumulate) {
    (void)xq16;
    qmx_matmul_tiled_scalar(out, xq, x_scales, w, accumulate);
}

static float scalar_dot_f32(const float * QMX_RESTRICT a,
                            const float * QMX_RESTRICT b,
                            uint32_t n) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

static void scalar_saxpy_f32(float       * QMX_RESTRICT y,
                             float a,
                             const float * QMX_RESTRICT x,
                             uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        y[i] += a * x[i];
    }
}

static int32_t scalar_dot_q8(const int8_t  * QMX_RESTRICT q8,
                             const int16_t * QMX_RESTRICT q16,
                             const int8_t  * QMX_RESTRICT k8,
                             uint32_t n) {
    (void)q16;
    int32_t sum = 0;
    for (uint32_t i = 0; i < n; i++) {
        sum += (int32_t)q8[i] * (int32_t)k8[i];
    }
    return sum;
}

const QmxBackendOps qmx_ops_scalar = {
    "Scalar",
    scalar_prepare_act,
    scalar_matmul_tiled,
    qmx_rmsnorm_scalar,
    scalar_dot_f32,
    scalar_saxpy_f32,
    scalar_dot_q8,
    qmx_rmsnorm_quant_scalar,
    qmx_swiglu_quant_scalar,
    qmx_f16_to_f32_scalar,
    qmx_f32_to_f16_scalar
};

static const QmxBackendOps * const g_builtin_backends[QMX_BACKEND_COUNT] = {
    [QMX_BACKEND_SCALAR] = &qmx_ops_scalar,
#if defined(QMX_ENABLE_BACKEND_P54C)
    [QMX_BACKEND_P54C] = &qmx_ops_p54c,
#endif
#if defined(QMX_ENABLE_BACKEND_MMX)
    [QMX_BACKEND_MMX] = &qmx_ops_mmx,
#endif
#if defined(QMX_ENABLE_BACKEND_3DNOW)
    [QMX_BACKEND_3DNOW] = &qmx_ops_3dnow,
#endif
#if defined(QMX_ENABLE_BACKEND_SSE2)
    [QMX_BACKEND_SSE2] = &qmx_ops_sse2,
#endif
#if defined(QMX_ENABLE_BACKEND_AVX2)
    [QMX_BACKEND_AVX2] = &qmx_ops_avx2,
#endif
#if defined(QMX_ENABLE_BACKEND_NEON)
    [QMX_BACKEND_NEON] = &qmx_ops_neon,
#endif
#if defined(QMX_ENABLE_BACKEND_ARM_DSP)
    [QMX_BACKEND_ARM_DSP] = &qmx_ops_arm_dsp,
#endif
#if defined(QMX_ENABLE_BACKEND_MCU32)
    [QMX_BACKEND_MCU32] = &qmx_ops_mcu32,
#endif
#if defined(QMX_ENABLE_BACKEND_X86_EXT)
    [QMX_BACKEND_SSE]    = &qmx_ops_sse,
    [QMX_BACKEND_SSE3]   = &qmx_ops_sse3,
    [QMX_BACKEND_SSSE3]  = &qmx_ops_ssse3,
    [QMX_BACKEND_SSE41]  = &qmx_ops_sse41,
    [QMX_BACKEND_SSE42]  = &qmx_ops_sse42,
    [QMX_BACKEND_AVX]    = &qmx_ops_avx,
    [QMX_BACKEND_AVX512] = &qmx_ops_avx512,
#endif
#if defined(QMX_ENABLE_BACKEND_CORTEX_M)
    [QMX_BACKEND_CORTEX_M0]  = &qmx_ops_cortex_m0,
    [QMX_BACKEND_CORTEX_M23] = &qmx_ops_cortex_m23,
    [QMX_BACKEND_CORTEX_M3]  = &qmx_ops_cortex_m3,
    [QMX_BACKEND_CORTEX_M4]  = &qmx_ops_cortex_m4,
    [QMX_BACKEND_CORTEX_M33] = &qmx_ops_cortex_m33,
    [QMX_BACKEND_CORTEX_M55] = &qmx_ops_cortex_m55,
#endif
#if defined(QMX_ENABLE_BACKEND_NEON_DOT)
    [QMX_BACKEND_NEON_DOT] = &qmx_ops_neon_dot,
#endif
};

#ifndef QMX_NO_CUSTOM_BACKENDS
static const QmxBackendOps *g_custom_ops[QMX_BACKEND_COUNT];
#endif

#if (defined(__i386__) || defined(__x86_64__)) && (defined(__GNUC__) || defined(__clang__))
#define QMX_HAS_X86_CPUID 1

#if defined(__i386__)
static int x86_has_cpuid(void) {
    uint32_t f1, f2;
    __asm__ volatile (
        "pushfl\n\t"
        "popl %0\n\t"
        "movl %0, %1\n\t"
        "xorl $0x200000, %0\n\t"
        "pushl %0\n\t"
        "popfl\n\t"
        "pushfl\n\t"
        "popl %0\n\t"
        "pushl %1\n\t"
        "popfl\n\t"
        : "=&r"(f1), "=&r"(f2)
        :
        : "cc"
    );
    return ((f1 ^ f2) & 0x200000u) != 0u;
}
#else
static int x86_has_cpuid(void) {
    return 1;
}
#endif

static void x86_cpuid(uint32_t leaf, uint32_t subleaf,
                      uint32_t *eax, uint32_t *ebx, uint32_t *ecx, uint32_t *edx) {
#if defined(__i386__)
    __asm__ volatile (
        "xchgl %%ebx, %1\n\t"
        "cpuid\n\t"
        "xchgl %%ebx, %1\n\t"
        : "=a"(*eax), "=&r"(*ebx), "=c"(*ecx), "=d"(*edx)
        : "0"(leaf), "2"(subleaf)
    );
#else
    __asm__ volatile (
        "cpuid"
        : "=a"(*eax), "=b"(*ebx), "=c"(*ecx), "=d"(*edx)
        : "0"(leaf), "2"(subleaf)
    );
#endif
}

static uint64_t x86_xgetbv0(void) {
    uint32_t eax, edx;
    __asm__ volatile (
        ".byte 0x0f, 0x01, 0xd0"
        : "=a"(eax), "=d"(edx)
        : "c"(0u)
    );
    return ((uint64_t)edx << 32) | (uint64_t)eax;
}

#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#define QMX_HAS_X86_CPUID 1

static int x86_has_cpuid(void) {
    return 1;
}

static void x86_cpuid(uint32_t leaf, uint32_t subleaf,
                      uint32_t *eax, uint32_t *ebx, uint32_t *ecx, uint32_t *edx) {
    int regs[4];
    __cpuidex(regs, (int)leaf, (int)subleaf);
    *eax = (uint32_t)regs[0];
    *ebx = (uint32_t)regs[1];
    *ecx = (uint32_t)regs[2];
    *edx = (uint32_t)regs[3];
}

static uint64_t x86_xgetbv0(void) {
    return (uint64_t)_xgetbv(0u);
}

#else
#define QMX_HAS_X86_CPUID 0
#endif

#if QMX_HAS_X86_CPUID
static int x86_leaf1_features(uint32_t *out_ecx, uint32_t *out_edx) {
    if (!x86_has_cpuid()) return 0;
    uint32_t eax, ebx, ecx, edx;
    x86_cpuid(0u, 0u, &eax, &ebx, &ecx, &edx);
    if (eax < 1u) return 0;
    x86_cpuid(1u, 0u, &eax, &ebx, out_ecx, out_edx);
    return 1;
}
#endif

int qmx_cpu_has_mmx(void) {
#if defined(__x86_64__) || defined(_M_X64)
    return 1;
#elif QMX_HAS_X86_CPUID
    uint32_t ecx = 0, edx = 0;
    if (!x86_leaf1_features(&ecx, &edx)) return 0;
    return (edx & (1u << 23)) ? 1 : 0;
#else
    return 0;
#endif
}

int qmx_cpu_has_3dnow(void) {
#if (defined(__i386__) || defined(_M_IX86)) && QMX_HAS_X86_CPUID
    if (!x86_has_cpuid()) return 0;
    uint32_t eax, ebx, ecx, edx;
    x86_cpuid(0x80000000u, 0u, &eax, &ebx, &ecx, &edx);
    if (eax < 0x80000001u) return 0;
    x86_cpuid(0x80000001u, 0u, &eax, &ebx, &ecx, &edx);
    return (edx & (1u << 31)) ? 1 : 0;
#else
    return 0;
#endif
}

int qmx_cpu_has_sse(void) {
#if defined(__x86_64__) || defined(_M_X64)
    return 1;
#elif QMX_HAS_X86_CPUID
    uint32_t ecx = 0, edx = 0;
    if (!x86_leaf1_features(&ecx, &edx)) return 0;
    return (edx & (1u << 25)) ? 1 : 0;
#else
    return 0;
#endif
}

int qmx_cpu_has_sse2(void) {
#if defined(__x86_64__) || defined(_M_X64)
    return 1;
#elif QMX_HAS_X86_CPUID
    uint32_t ecx = 0, edx = 0;
    if (!x86_leaf1_features(&ecx, &edx)) return 0;
    return (edx & (1u << 26)) ? 1 : 0;
#else
    return 0;
#endif
}

int qmx_cpu_has_sse3(void) {
#if QMX_HAS_X86_CPUID
    uint32_t ecx = 0, edx = 0;
    if (!x86_leaf1_features(&ecx, &edx)) return 0;
    return (ecx & (1u << 0)) ? 1 : 0;
#else
    return 0;
#endif
}

int qmx_cpu_has_ssse3(void) {
#if QMX_HAS_X86_CPUID
    uint32_t ecx = 0, edx = 0;
    if (!x86_leaf1_features(&ecx, &edx)) return 0;
    return (ecx & (1u << 9)) ? 1 : 0;
#else
    return 0;
#endif
}

int qmx_cpu_has_sse41(void) {
#if QMX_HAS_X86_CPUID
    uint32_t ecx = 0, edx = 0;
    if (!x86_leaf1_features(&ecx, &edx)) return 0;
    return (ecx & (1u << 19)) ? 1 : 0;
#else
    return 0;
#endif
}

int qmx_cpu_has_sse42(void) {
#if QMX_HAS_X86_CPUID
    uint32_t ecx = 0, edx = 0;
    if (!x86_leaf1_features(&ecx, &edx)) return 0;
    uint32_t mask = (1u << 20) | (1u << 23);
    return ((ecx & mask) == mask) ? 1 : 0;
#else
    return 0;
#endif
}

int qmx_cpu_has_avx(void) {
#if QMX_HAS_X86_CPUID
    uint32_t ecx = 0, edx = 0;
    if (!x86_leaf1_features(&ecx, &edx)) return 0;
    uint32_t osxsave_avx = (1u << 27) | (1u << 28);
    if ((ecx & osxsave_avx) != osxsave_avx) return 0;
    return ((x86_xgetbv0() & 0x6u) == 0x6u) ? 1 : 0;
#else
    return 0;
#endif
}

int qmx_cpu_has_avx2(void) {
#if QMX_HAS_X86_CPUID
    if (!qmx_cpu_has_avx()) return 0;
    uint32_t ecx1 = 0, edx1 = 0;
    if (!x86_leaf1_features(&ecx1, &edx1)) return 0;
    uint32_t need_l1 = (1u << 12) | (1u << 29);
    if ((ecx1 & need_l1) != need_l1) return 0;
    uint32_t eax, ebx, ecx, edx;
    x86_cpuid(0u, 0u, &eax, &ebx, &ecx, &edx);
    if (eax < 7u) return 0;
    x86_cpuid(7u, 0u, &eax, &ebx, &ecx, &edx);
    return (ebx & (1u << 5)) ? 1 : 0;
#else
    return 0;
#endif
}

int qmx_cpu_has_avx512(void) {
#if QMX_HAS_X86_CPUID
    uint32_t ecx1 = 0, edx1 = 0;
    if (!x86_leaf1_features(&ecx1, &edx1)) return 0;
    if ((ecx1 & ((1u << 12) | (1u << 27) | (1u << 29))) != ((1u << 12) | (1u << 27) | (1u << 29))) return 0;
    if ((x86_xgetbv0() & 0xE6u) != 0xE6u) return 0;
    uint32_t eax, ebx, ecx, edx;
    x86_cpuid(0u, 0u, &eax, &ebx, &ecx, &edx);
    if (eax < 7u) return 0;
    x86_cpuid(7u, 0u, &eax, &ebx, &ecx, &edx);
    uint32_t need_ebx = (1u << 16) | (1u << 30) | (1u << 31);
    uint32_t need_ecx = (1u << 11);
    return (((ebx & need_ebx) == need_ebx) && ((ecx & need_ecx) == need_ecx)) ? 1 : 0;
#else
    return 0;
#endif
}

int qmx_cpu_has_neon(void) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return 1;
#elif defined(__arm__)
#if defined(__linux__)
    int fd = open("/proc/self/auxv", O_RDONLY);
    if (fd >= 0) {
        uint32_t pair[2];
        int found = 0;
        while (read(fd, pair, sizeof(pair)) == (ssize_t)sizeof(pair)) {
            if (pair[0] == 0u) break;
            if (pair[0] == 16u) {
                found = ((pair[1] & (1u << 12)) != 0u) ? 1 : 0;
                break;
            }
        }
        close(fd);
        return found;
    }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return 1;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int qmx_cpu_has_neon_dot(void) {
#if defined(__ARM_FEATURE_DOTPROD) || (defined(__APPLE__) && defined(__aarch64__))
    return 1;
#elif defined(__linux__) && defined(__aarch64__)
    int fd = open("/proc/self/auxv", O_RDONLY);
    if (fd >= 0) {
        uint64_t pair[2];
        int found = 0;
        while (read(fd, pair, sizeof(pair)) == (ssize_t)sizeof(pair)) {
            if (pair[0] == 0u) break;
            if (pair[0] == 16u) {
                found = ((pair[1] & (1ULL << 20)) != 0ULL) ? 1 : 0;
                break;
            }
        }
        close(fd);
        return found;
    }
    return 0;
#else
    return 0;
#endif
}

int qmx_cpu_has_arm_dsp(void) {
#if defined(__ARM_FEATURE_DSP)
    return 1;
#else
    return 0;
#endif
}

void qmx_register_backend(QmxBackend backend, const QmxBackendOps *ops) {
#ifndef QMX_NO_CUSTOM_BACKENDS
    uint32_t idx = (uint32_t)backend;
    if (idx < (uint32_t)QMX_BACKEND_COUNT && ops != NULL) {
        g_custom_ops[idx] = ops;
    }
#else
    (void)backend;
    (void)ops;
#endif
}

const QmxBackendOps *qmx_get_backend_ops(QmxBackend backend) {
    uint32_t idx = (uint32_t)backend;
    if (idx < (uint32_t)QMX_BACKEND_COUNT) {
#ifndef QMX_NO_CUSTOM_BACKENDS
        if (g_custom_ops[idx] != NULL) return g_custom_ops[idx];
#endif
        if (g_builtin_backends[idx] != NULL) return g_builtin_backends[idx];
    }
    return &qmx_ops_scalar;
}

static int backend_is_supported(QmxBackend b) {
    if (b == QMX_BACKEND_SCALAR) return 1;
    if (qmx_get_backend_ops(b) == &qmx_ops_scalar) return 0;
    switch (b) {
        case QMX_BACKEND_P54C:
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
            return 1;
#else
            return 0;
#endif
        case QMX_BACKEND_MMX:        return qmx_cpu_has_mmx();
        case QMX_BACKEND_3DNOW:      return qmx_cpu_has_3dnow();
        case QMX_BACKEND_SSE:        return qmx_cpu_has_sse();
        case QMX_BACKEND_SSE2:       return qmx_cpu_has_sse2();
        case QMX_BACKEND_SSE3:       return qmx_cpu_has_sse3();
        case QMX_BACKEND_SSSE3:      return qmx_cpu_has_ssse3();
        case QMX_BACKEND_SSE41:      return qmx_cpu_has_sse41();
        case QMX_BACKEND_SSE42:      return qmx_cpu_has_sse42();
        case QMX_BACKEND_AVX:        return qmx_cpu_has_avx();
        case QMX_BACKEND_AVX2:       return qmx_cpu_has_avx2();
        case QMX_BACKEND_AVX512:     return qmx_cpu_has_avx512();
        case QMX_BACKEND_NEON:       return qmx_cpu_has_neon();
        case QMX_BACKEND_NEON_DOT:   return qmx_cpu_has_neon_dot();
        case QMX_BACKEND_ARM_DSP:    return qmx_cpu_has_arm_dsp();
        case QMX_BACKEND_MCU32:
        case QMX_BACKEND_CORTEX_M0:
        case QMX_BACKEND_CORTEX_M23:
        case QMX_BACKEND_CORTEX_M3:
        case QMX_BACKEND_CORTEX_M4:
        case QMX_BACKEND_CORTEX_M33:
        case QMX_BACKEND_CORTEX_M55: return 1;
        default:                     return 0;
    }
}

QmxBackend qmx_detect_backend(void) {
    if (qmx_cpu_has_avx512() && qmx_get_backend_ops(QMX_BACKEND_AVX512) != &qmx_ops_scalar) return QMX_BACKEND_AVX512;
    if (qmx_cpu_has_avx2()   && qmx_get_backend_ops(QMX_BACKEND_AVX2)   != &qmx_ops_scalar) return QMX_BACKEND_AVX2;
    if (qmx_cpu_has_avx()    && qmx_get_backend_ops(QMX_BACKEND_AVX)    != &qmx_ops_scalar) return QMX_BACKEND_AVX;
    if (qmx_cpu_has_sse42()  && qmx_get_backend_ops(QMX_BACKEND_SSE42)  != &qmx_ops_scalar) return QMX_BACKEND_SSE42;
    if (qmx_cpu_has_sse41()  && qmx_get_backend_ops(QMX_BACKEND_SSE41)  != &qmx_ops_scalar) return QMX_BACKEND_SSE41;
    if (qmx_cpu_has_ssse3()  && qmx_get_backend_ops(QMX_BACKEND_SSSE3)  != &qmx_ops_scalar) return QMX_BACKEND_SSSE3;
    if (qmx_cpu_has_sse3()   && qmx_get_backend_ops(QMX_BACKEND_SSE3)   != &qmx_ops_scalar) return QMX_BACKEND_SSE3;
    if (qmx_cpu_has_sse2()   && qmx_get_backend_ops(QMX_BACKEND_SSE2)   != &qmx_ops_scalar) return QMX_BACKEND_SSE2;
    if (qmx_cpu_has_sse()    && qmx_get_backend_ops(QMX_BACKEND_SSE)    != &qmx_ops_scalar) return QMX_BACKEND_SSE;
    if (qmx_cpu_has_3dnow()  && qmx_get_backend_ops(QMX_BACKEND_3DNOW)  != &qmx_ops_scalar) return QMX_BACKEND_3DNOW;
    if (qmx_cpu_has_mmx()    && qmx_get_backend_ops(QMX_BACKEND_MMX)    != &qmx_ops_scalar) return QMX_BACKEND_MMX;
    if (qmx_cpu_has_neon_dot() && qmx_get_backend_ops(QMX_BACKEND_NEON_DOT) != &qmx_ops_scalar) return QMX_BACKEND_NEON_DOT;
    if (qmx_cpu_has_neon()   && qmx_get_backend_ops(QMX_BACKEND_NEON)   != &qmx_ops_scalar) return QMX_BACKEND_NEON;
#if defined(__ARM_FEATURE_MVE)
    if (qmx_get_backend_ops(QMX_BACKEND_CORTEX_M55) != &qmx_ops_scalar) return QMX_BACKEND_CORTEX_M55;
#endif
#if defined(__ARM_ARCH_8M_MAIN__) && defined(__ARM_FEATURE_DSP)
    if (qmx_get_backend_ops(QMX_BACKEND_CORTEX_M33) != &qmx_ops_scalar) return QMX_BACKEND_CORTEX_M33;
#endif
#if defined(__ARM_ARCH_7EM__) && defined(__ARM_FEATURE_DSP)
    if (qmx_get_backend_ops(QMX_BACKEND_CORTEX_M4) != &qmx_ops_scalar) return QMX_BACKEND_CORTEX_M4;
#endif
    if (qmx_cpu_has_arm_dsp() && qmx_get_backend_ops(QMX_BACKEND_ARM_DSP) != &qmx_ops_scalar) return QMX_BACKEND_ARM_DSP;
#if defined(__ARM_ARCH_7M__)
    if (qmx_get_backend_ops(QMX_BACKEND_CORTEX_M3) != &qmx_ops_scalar) return QMX_BACKEND_CORTEX_M3;
#endif
#if defined(__ARM_ARCH_8M_BASE__)
    if (qmx_get_backend_ops(QMX_BACKEND_CORTEX_M23) != &qmx_ops_scalar) return QMX_BACKEND_CORTEX_M23;
#endif
#if defined(__ARM_ARCH_6M__)
    if (qmx_get_backend_ops(QMX_BACKEND_CORTEX_M0) != &qmx_ops_scalar) return QMX_BACKEND_CORTEX_M0;
#endif
#if defined(__riscv) || defined(__xtensa__)
    if (qmx_get_backend_ops(QMX_BACKEND_MCU32) != &qmx_ops_scalar) return QMX_BACKEND_MCU32;
#elif defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    if (qmx_get_backend_ops(QMX_BACKEND_P54C) != &qmx_ops_scalar) return QMX_BACKEND_P54C;
#endif
    return QMX_BACKEND_SCALAR;
}

const char *qmx_backend_name(QmxBackend backend) {
    switch (backend) {
        case QMX_BACKEND_SCALAR:     return "Scalar";
        case QMX_BACKEND_P54C:       return "P54C";
        case QMX_BACKEND_MMX:        return "MMX";
        case QMX_BACKEND_3DNOW:      return "3DNow!";
        case QMX_BACKEND_SSE:        return "SSE";
        case QMX_BACKEND_SSE2:       return "SSE2";
        case QMX_BACKEND_SSE3:       return "SSE3";
        case QMX_BACKEND_SSSE3:      return "SSSE3";
        case QMX_BACKEND_SSE41:      return "SSE4.1";
        case QMX_BACKEND_SSE42:      return "SSE4.2";
        case QMX_BACKEND_AVX:        return "AVX";
        case QMX_BACKEND_AVX2:       return "AVX2";
        case QMX_BACKEND_AVX512:     return "AVX-512";
        case QMX_BACKEND_NEON:       return "NEON";
        case QMX_BACKEND_NEON_DOT:   return "NEON-Dot";
        case QMX_BACKEND_ARM_DSP:    return "ARM-DSP";
        case QMX_BACKEND_MCU32:      return "MCU32";
        case QMX_BACKEND_CORTEX_M0:  return "Cortex-M0/M0+";
        case QMX_BACKEND_CORTEX_M23: return "Cortex-M23";
        case QMX_BACKEND_CORTEX_M3:  return "Cortex-M3";
        case QMX_BACKEND_CORTEX_M4:  return "Cortex-M4/M7";
        case QMX_BACKEND_CORTEX_M33: return "Cortex-M33";
        case QMX_BACKEND_CORTEX_M55: return "Cortex-M55";
        case QMX_BACKEND_AUTO:       return "Auto";
        default:                     return "Unknown";
    }
}

int qmx_backend_needs_xq16(QmxBackend backend) {
    switch (backend) {
        case QMX_BACKEND_MMX:
        case QMX_BACKEND_3DNOW:
        case QMX_BACKEND_SSE:
        case QMX_BACKEND_SSE2:
        case QMX_BACKEND_SSE3:
        case QMX_BACKEND_ARM_DSP:
        case QMX_BACKEND_CORTEX_M4:
        case QMX_BACKEND_CORTEX_M33:
            return 1;
        default:
            return 0;
    }
}

void qmx_bind_backend_ops(QmxModel *model, QmxBackend requested) {
    if (!model) return;
    QmxBackend chosen = requested;
    if (chosen == QMX_BACKEND_AUTO || !backend_is_supported(chosen)) {
        chosen = qmx_detect_backend();
    }
    if (model->state.slots[0] != NULL && model->state.xq16 == NULL && qmx_backend_needs_xq16(chosen)) {
        chosen = QMX_BACKEND_SCALAR;
    }
    model->backend = chosen;
    model->ops = qmx_get_backend_ops(chosen);
}
