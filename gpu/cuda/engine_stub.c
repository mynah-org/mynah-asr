/* gpu/cuda/engine_stub.c — the cuda engine in a build WITHOUT nvcc: it says
 * so, in the words the CI job greps for, and refuses. Never linked together
 * with engine.cu. */
#include "../asr_engine.h"

#include <stdio.h>

asr_engine *asr_engine_open_cuda(const asr_engine_cfg *cfg, char *err, size_t errcap) {
    (void)cfg;
    snprintf(err, errcap, "the cuda engine is not compiled into this binary (built without nvcc: make -C gpu cpu)");
    return NULL;
}
