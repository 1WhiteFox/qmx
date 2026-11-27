/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/qmx_thread.h"
#include "../include/qmx_runtime.h"
#ifndef QMX_NO_MALLOC
#include <stdlib.h>
#endif
#include <string.h>

#define QMX_SPIN_ITERS 4096

#if defined(QMX_ENABLE_THREADS) && !defined(QMX_NO_THREADS) && !defined(QMX_NO_MALLOC) && !defined(__DJGPP__)
    #if defined(_WIN32)
        #define QMX_THREAD_WIN32 1
        #define QMX_THREAD_POSIX 0
        #include <windows.h>
        #if defined(_MSC_VER)
            #include <intrin.h>
        #endif
    #elif defined(__unix__) || defined(__APPLE__) || defined(__linux__)
        #define QMX_THREAD_WIN32 0
        #define QMX_THREAD_POSIX 1
        #include <pthread.h>
    #else
        #define QMX_THREAD_WIN32 0
        #define QMX_THREAD_POSIX 0
    #endif
#else
    #define QMX_THREAD_WIN32 0
    #define QMX_THREAD_POSIX 0
#endif

#if QMX_THREAD_POSIX || QMX_THREAD_WIN32
QMX_INLINE void qmx_cpu_relax(void) {
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
    _mm_pause();
#elif defined(__i386__) || defined(__x86_64__)
    __asm__ volatile ("pause" ::: "memory");
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ volatile ("yield" ::: "memory");
#else
    __asm__ volatile ("" ::: "memory");
#endif
}
#endif

#if QMX_THREAD_POSIX

struct QmxThreadPool;

typedef struct {
    struct QmxThreadPool *pool;
    int                   tid;
} QmxWorkerArg;

struct QmxThreadPool {
    pthread_t       workers[QMX_MAX_THREADS];
    QmxWorkerArg    args[QMX_MAX_THREADS];
    int             num_threads;
    volatile int    shutdown;
    volatile uint32_t gen;
    volatile int    done_count;
    volatile int    sleeping;
    QmxJobFn        job_fn;
    void           *job_ctx;
    int             active_threads;
    volatile int    main_waiting;
    pthread_mutex_t mtx;
    pthread_cond_t  cv;
    pthread_cond_t  done_cv;
};

static void *qmx_posix_worker(void *raw) {
    QmxWorkerArg *wa = (QmxWorkerArg *)raw;
    QmxThreadPool *p = wa->pool;
    int tid = wa->tid;
    uint32_t seen_gen = 0u;

    for (;;) {
        uint32_t cur_gen = __atomic_load_n(&p->gen, __ATOMIC_ACQUIRE);
        if (cur_gen == seen_gen) {
            if (__atomic_load_n(&p->shutdown, __ATOMIC_ACQUIRE)) break;
            int woke = 0;
            for (int s = 0; s < QMX_SPIN_ITERS; s++) {
                qmx_cpu_relax();
                cur_gen = __atomic_load_n(&p->gen, __ATOMIC_ACQUIRE);
                if (cur_gen != seen_gen || __atomic_load_n(&p->shutdown, __ATOMIC_ACQUIRE)) {
                    woke = 1;
                    break;
                }
            }
            if (!woke) {
                pthread_mutex_lock(&p->mtx);
                __atomic_add_fetch(&p->sleeping, 1, __ATOMIC_SEQ_CST);
                while (__atomic_load_n(&p->gen, __ATOMIC_SEQ_CST) == seen_gen &&
                       !__atomic_load_n(&p->shutdown, __ATOMIC_SEQ_CST)) {
                    pthread_cond_wait(&p->cv, &p->mtx);
                }
                __atomic_sub_fetch(&p->sleeping, 1, __ATOMIC_SEQ_CST);
                pthread_mutex_unlock(&p->mtx);
                if (__atomic_load_n(&p->shutdown, __ATOMIC_ACQUIRE)) break;
                cur_gen = __atomic_load_n(&p->gen, __ATOMIC_ACQUIRE);
                if (cur_gen == seen_gen) continue;
            } else {
                if (__atomic_load_n(&p->shutdown, __ATOMIC_ACQUIRE)) break;
            }
        }

        seen_gen = cur_gen;
        QmxJobFn fn = p->job_fn;
        void *ctx = p->job_ctx;
        int active = p->active_threads;
        if (fn && tid < active) {
            fn(ctx, tid, active);
        }
        int finished = __atomic_add_fetch(&p->done_count, 1, __ATOMIC_SEQ_CST);
        if (finished >= p->num_threads - 1 && __atomic_load_n(&p->main_waiting, __ATOMIC_SEQ_CST)) {
            pthread_mutex_lock(&p->mtx);
            pthread_cond_signal(&p->done_cv);
            pthread_mutex_unlock(&p->mtx);
        }
    }
    return NULL;
}

QmxThreadPool *qmx_pool_create(int num_threads) {
    if (num_threads <= 1) return NULL;
    if (num_threads > QMX_MAX_THREADS) num_threads = QMX_MAX_THREADS;
    QmxThreadPool *p = (QmxThreadPool *)calloc(1, sizeof(QmxThreadPool));
    if (!p) return NULL;
    pthread_mutex_init(&p->mtx, NULL);
    pthread_cond_init(&p->cv, NULL);
    pthread_cond_init(&p->done_cv, NULL);
    p->num_threads = 1;

    for (int t = 1; t < num_threads; t++) {
        p->args[t].pool = p;
        p->args[t].tid = t;
        if (pthread_create(&p->workers[t], NULL, qmx_posix_worker, &p->args[t]) == 0) {
            p->num_threads = t + 1;
        } else {
            break;
        }
    }
    if (p->num_threads <= 1) {
        pthread_cond_destroy(&p->done_cv);
        pthread_cond_destroy(&p->cv);
        pthread_mutex_destroy(&p->mtx);
        free(p);
        return NULL;
    }
    return p;
}

void qmx_pool_destroy(QmxThreadPool *pool) {
    if (!pool) return;
    pthread_mutex_lock(&pool->mtx);
    __atomic_store_n(&pool->shutdown, 1, __ATOMIC_SEQ_CST);
    pthread_cond_broadcast(&pool->cv);
    pthread_mutex_unlock(&pool->mtx);

    for (int t = 1; t < pool->num_threads; t++) {
        pthread_join(pool->workers[t], NULL);
    }
    pthread_cond_destroy(&pool->done_cv);
    pthread_cond_destroy(&pool->cv);
    pthread_mutex_destroy(&pool->mtx);
    free(pool);
}

int qmx_pool_size(const QmxThreadPool *pool) {
    return pool ? pool->num_threads : 1;
}

void qmx_pool_run(QmxThreadPool *pool, int active_threads, QmxJobFn fn, void *ctx) {
    if (!fn) return;
    if (!pool || active_threads <= 1 || pool->num_threads <= 1) {
        fn(ctx, 0, 1);
        return;
    }
    if (active_threads > pool->num_threads) active_threads = pool->num_threads;

    pool->job_fn = fn;
    pool->job_ctx = ctx;
    pool->active_threads = active_threads;
    __atomic_store_n(&pool->done_count, 0, __ATOMIC_RELAXED);
    __atomic_add_fetch(&pool->gen, 1u, __ATOMIC_SEQ_CST);

    if (__atomic_load_n(&pool->sleeping, __ATOMIC_SEQ_CST) > 0) {
        pthread_mutex_lock(&pool->mtx);
        pthread_cond_broadcast(&pool->cv);
        pthread_mutex_unlock(&pool->mtx);
    }

    fn(ctx, 0, active_threads);

    int need = pool->num_threads - 1;
    int spin = 0;
    while (__atomic_load_n(&pool->done_count, __ATOMIC_ACQUIRE) < need) {
        if (spin < QMX_SPIN_ITERS) {
            qmx_cpu_relax();
            spin++;
        } else {
            pthread_mutex_lock(&pool->mtx);
            __atomic_store_n(&pool->main_waiting, 1, __ATOMIC_SEQ_CST);
            while (__atomic_load_n(&pool->done_count, __ATOMIC_SEQ_CST) < need) {
                pthread_cond_wait(&pool->done_cv, &pool->mtx);
            }
            __atomic_store_n(&pool->main_waiting, 0, __ATOMIC_SEQ_CST);
            pthread_mutex_unlock(&pool->mtx);
            break;
        }
    }
}

#elif QMX_THREAD_WIN32

struct QmxThreadPool;

typedef struct {
    struct QmxThreadPool *pool;
    int                   tid;
} QmxWorkerArg;

struct QmxThreadPool {
    HANDLE        workers[QMX_MAX_THREADS];
    HANDLE        wake_events[QMX_MAX_THREADS];
    HANDLE        done_event;
    volatile LONG worker_sleeping[QMX_MAX_THREADS];
    volatile LONG main_waiting;
    QmxWorkerArg  args[QMX_MAX_THREADS];
    int           num_threads;
    volatile LONG shutdown;
    volatile LONG gen;
    volatile LONG done_count;
    QmxJobFn      job_fn;
    void         *job_ctx;
    int           active_threads;
};

static DWORD WINAPI qmx_win32_worker(LPVOID raw) {
    QmxWorkerArg *wa = (QmxWorkerArg *)raw;
    QmxThreadPool *p = wa->pool;
    int tid = wa->tid;
    LONG seen_gen = 0;

    for (;;) {
        LONG cur_gen = InterlockedCompareExchange(&p->gen, 0, 0);
        if (cur_gen == seen_gen) {
            if (InterlockedCompareExchange(&p->shutdown, 0, 0)) break;
            int woke = 0;
            for (int s = 0; s < QMX_SPIN_ITERS; s++) {
                qmx_cpu_relax();
                cur_gen = InterlockedCompareExchange(&p->gen, 0, 0);
                if (cur_gen != seen_gen || InterlockedCompareExchange(&p->shutdown, 0, 0)) {
                    woke = 1;
                    break;
                }
            }
            if (!woke) {
                InterlockedExchange(&p->worker_sleeping[tid], 1);
                cur_gen = InterlockedCompareExchange(&p->gen, 0, 0);
                if (cur_gen == seen_gen && !InterlockedCompareExchange(&p->shutdown, 0, 0)) {
                    WaitForSingleObject(p->wake_events[tid], INFINITE);
                }
                InterlockedExchange(&p->worker_sleeping[tid], 0);
                if (InterlockedCompareExchange(&p->shutdown, 0, 0)) break;
                cur_gen = InterlockedCompareExchange(&p->gen, 0, 0);
                if (cur_gen == seen_gen) continue;
            } else {
                if (InterlockedCompareExchange(&p->shutdown, 0, 0)) break;
            }
        }

        seen_gen = cur_gen;
        QmxJobFn fn = p->job_fn;
        void *ctx = p->job_ctx;
        int active = p->active_threads;
        if (fn && tid < active) {
            fn(ctx, tid, active);
        }
        LONG finished = InterlockedIncrement(&p->done_count);
        if (finished >= (LONG)(p->num_threads - 1)) {
            if (InterlockedCompareExchange(&p->main_waiting, 0, 1) == 1 && p->done_event) {
                SetEvent(p->done_event);
            }
        }
    }
    return 0;
}

QmxThreadPool *qmx_pool_create(int num_threads) {
    if (num_threads <= 1) return NULL;
    if (num_threads > QMX_MAX_THREADS) num_threads = QMX_MAX_THREADS;
    QmxThreadPool *p = (QmxThreadPool *)calloc(1, sizeof(QmxThreadPool));
    if (!p) return NULL;
    p->done_event = CreateEventA(NULL, FALSE, FALSE, NULL);
    if (!p->done_event) {
        free(p);
        return NULL;
    }
    p->num_threads = 1;

    for (int t = 1; t < num_threads; t++) {
        p->wake_events[t] = CreateEventA(NULL, FALSE, FALSE, NULL);
        if (!p->wake_events[t]) break;
        p->args[t].pool = p;
        p->args[t].tid = t;
        p->workers[t] = CreateThread(NULL, 0, qmx_win32_worker, &p->args[t], 0, NULL);
        if (p->workers[t]) {
            p->num_threads = t + 1;
        } else {
            CloseHandle(p->wake_events[t]);
            p->wake_events[t] = NULL;
            break;
        }
    }
    if (p->num_threads <= 1) {
        if (p->done_event) CloseHandle(p->done_event);
        free(p);
        return NULL;
    }
    return p;
}

void qmx_pool_destroy(QmxThreadPool *pool) {
    if (!pool) return;
    InterlockedExchange(&pool->shutdown, 1);
    for (int t = 1; t < pool->num_threads; t++) {
        if (pool->wake_events[t]) SetEvent(pool->wake_events[t]);
    }
    for (int t = 1; t < pool->num_threads; t++) {
        if (pool->workers[t]) {
            WaitForSingleObject(pool->workers[t], INFINITE);
            CloseHandle(pool->workers[t]);
        }
        if (pool->wake_events[t]) {
            CloseHandle(pool->wake_events[t]);
        }
    }
    if (pool->done_event) CloseHandle(pool->done_event);
    free(pool);
}

int qmx_pool_size(const QmxThreadPool *pool) {
    return pool ? pool->num_threads : 1;
}

void qmx_pool_run(QmxThreadPool *pool, int active_threads, QmxJobFn fn, void *ctx) {
    if (!fn) return;
    if (!pool || active_threads <= 1 || pool->num_threads <= 1) {
        fn(ctx, 0, 1);
        return;
    }
    if (active_threads > pool->num_threads) active_threads = pool->num_threads;

    pool->job_fn = fn;
    pool->job_ctx = ctx;
    pool->active_threads = active_threads;
    InterlockedExchange(&pool->done_count, 0);
    InterlockedIncrement(&pool->gen);

    for (int t = 1; t < pool->num_threads; t++) {
        if (InterlockedCompareExchange(&pool->worker_sleeping[t], 0, 1) == 1) {
            SetEvent(pool->wake_events[t]);
        }
    }

    fn(ctx, 0, active_threads);

    LONG need = (LONG)(pool->num_threads - 1);
    int spin = 0;
    while (InterlockedCompareExchange(&pool->done_count, 0, 0) < need) {
        if (spin < QMX_SPIN_ITERS) {
            qmx_cpu_relax();
            spin++;
        } else {
            InterlockedExchange(&pool->main_waiting, 1);
            if (InterlockedCompareExchange(&pool->done_count, 0, 0) < need) {
                WaitForSingleObject(pool->done_event, INFINITE);
            } else {
                InterlockedExchange(&pool->main_waiting, 0);
            }
        }
    }
}

#else

QmxThreadPool *qmx_pool_create(int num_threads) {
    (void)num_threads;
    return NULL;
}

void qmx_pool_destroy(QmxThreadPool *pool) {
    (void)pool;
}

int qmx_pool_size(const QmxThreadPool *pool) {
    (void)pool;
    return 1;
}

void qmx_pool_run(QmxThreadPool *pool, int active_threads, QmxJobFn fn, void *ctx) {
    (void)pool;
    (void)active_threads;
    if (fn) fn(ctx, 0, 1);
}

#endif