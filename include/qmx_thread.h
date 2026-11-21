#ifndef QMX_THREAD_H
#define QMX_THREAD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QMX_MAX_THREADS 16

typedef struct QmxThreadPool QmxThreadPool;
typedef void (*QmxJobFn)(void *ctx, int tid, int n_threads);

QmxThreadPool *qmx_pool_create(int num_threads);
void           qmx_pool_destroy(QmxThreadPool *pool);
int            qmx_pool_size(const QmxThreadPool *pool);
void           qmx_pool_run(QmxThreadPool *pool, int active_threads, QmxJobFn fn, void *ctx);

#ifdef __cplusplus
}
#endif

#endif