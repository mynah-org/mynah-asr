/* gpu/cuda/aed_stub.c — the GPU AED engine in a build WITHOUT it: it says so,
 * in the words the server maps to exit 78, and refuses. Never linked together
 * with aed.cu. */
#include "../aed_gpu.h"

#include <stdio.h>
#include <string.h>

asr_aed_gpu *asr_aed_gpu_open(const asr_aed_gpu_cfg *cfg, char *err, size_t errcap) {
    (void)cfg;
    snprintf(err, errcap, "the cuda AED engine is not compiled into this binary (built without nvcc: make -C gpu cpu)");
    return NULL;
}
void asr_aed_gpu_close(asr_aed_gpu *e) { (void)e; }
void asr_aed_gpu_offload(asr_aed_gpu *e, mynah_asr_offload *out) { (void)e; memset(out, 0, sizeof(*out)); }
void asr_aed_gpu_get_facts(const asr_aed_gpu *e, asr_aed_gpu_facts *f) { (void)e; memset(f, 0, sizeof(*f)); }
void asr_aed_gpu_get_stats(const asr_aed_gpu *e, asr_aed_gpu_stats *s) { (void)e; memset(s, 0, sizeof(*s)); }
int asr_aed_gpu_dead(const asr_aed_gpu *e) { (void)e; return 0; }
const char *asr_aed_gpu_error(const asr_aed_gpu *e) { (void)e; return ""; }
size_t asr_aed_gpu_dispatch_map(const asr_aed_gpu *e, char *buf, size_t cap) { (void)e; if (cap) buf[0] = '\0'; return 0; }
