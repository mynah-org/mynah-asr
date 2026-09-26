/* gpu/asr_engine.h — the one seam between the GPU server and whatever computes.
 *
 * The GPU server (gpu/server/) owns sockets, slots, books and the cohort
 * scheduler; an ENGINE owns everything after the PCM: the per-slot mel front
 * end, the resident model, the encoder step over a cohort, the greedy decode and
 * the detokenised text. Two engines implement this interface:
 *
 *   cuda   gpu/cuda/engine.cu   the product: weights and every slot's state
 *                               resident on one GPU, one batched step per cohort.
 *   cpu    gpu/engine_cpu.c     the REFERENCE: the library's own stream API
 *                               (src/mynah_asr.h) behind the same seam, so the
 *                               server can be tested on a machine without a GPU
 *                               and so the GPU's transcripts have an in-process
 *                               reference. Chosen ONLY by `--engine cpu`; it is
 *                               never a fallback the cuda engine degrades into.
 *
 * Contract of a step (asr_engine_step): for every requested slot the engine
 * consumes AT MOST one encoder chunk of that slot's buffered mel -- the chunk
 * the caller saw as ready -- or, when `finalize` is set and less than a chunk is
 * left, the padded tail. It returns the text appended by that chunk (bytes owned
 * by the engine, valid until the slot's next step or reset) and whether the
 * slot's utterance is now finished. A slot that is neither ready nor finalizing
 * produces no text and is not an error. The engine never touches a socket and
 * never blocks on one.
 *
 * Threading: ONE thread at a time per engine, the server's engine thread. The
 * ingest threads never call in: they fill the server's own per-slot PCM ring,
 * and the engine thread stages `asr_engine_slot_need_samples` of it into the
 * engine right before a step, exactly as the CPU scheduler stages into the
 * library (server/sched.c, sched_stage). So a slot never holds more than about
 * one chunk of mel inside the engine and the engine allocates nothing per step.
 *
 * Dispatch: an engine is a struct whose first member is `asr_engine` (the ops
 * table); the inline functions below are the only way the server calls one. */
#ifndef MYNAH_ASR_GPU_ENGINE_H
#define MYNAH_ASR_GPU_ENGINE_H

#include <stddef.h>

typedef struct asr_engine asr_engine;

typedef struct {
    const char *model_dir;   /* a converted pack (mynah.json + weights) */
    int cap;                 /* slots resident at start; the admission cap */
    int device;              /* cuda: the device ordinal; cpu: ignored */
    const char *precision;   /* cuda: "f32" (default) -- bf16/int8 are S14-8 */
    const char *gemm;        /* cuda: "own" (default, row-stable by construction)
                                or "cublas" (the comparison arm, measured) */
    int threads;             /* cpu: pool threads for the reference engine */
} asr_engine_cfg;

/* What one step produced for one requested slot. */
typedef struct {
    const char *text;        /* bytes appended by this step ("" when none) */
    double t0, t1;           /* the audio window (seconds) the text covers:
                                t0 = the previous step's t1, t1 = audio fed so
                                far -- the same partition the CPU server emits */
    int n_tokens;            /* tokens emitted by this step */
    int finished;            /* the padded tail was processed: `done` is due */
    int stepped;             /* an encoder chunk (or the tail) ran for this slot */
} asr_step_out;

typedef struct {
    int slot;
    int finalize;            /* flush everything left, tail included */
} asr_step_req;

/* Facts the server prints in its banner, /v1/health and --dispatch-map: every
 * one of them is what the engine RESOLVED, never what was asked. */
typedef struct {
    const char *name;        /* "cuda" | "cpu" */
    const char *device;      /* GPU name, or the CPU provider */
    const char *precision;   /* "f32" ... */
    const char *gemm;        /* "own-rowstable" | "cublas" | "sgemm-own" ... */
    const char *model_name;
    int cap, qmax, n_lookaheads, lookaheads[8], default_lookahead;
    int sample_rate, n_mels;
    double frame_sec;        /* one encoder frame, seconds */
    size_t vram_total, vram_used, vram_arena, vram_weights;  /* bytes; 0 on cpu */
    int graphs;              /* CUDA graphs in use (S14-8), 0 in phase 1 */
} asr_engine_facts;

/* Per-step counters the engine keeps, for the SIGUSR1 dump and /metrics. */
typedef struct {
    unsigned long steps;             /* cohort steps run */
    unsigned long rows;              /* encoder rows (frames) stacked, total */
    unsigned long lanes;             /* slots stepped, total */
    unsigned long decode_iters;      /* label-loop iterations, total */
    double step_wall_ms_sum;         /* wall inside asr_engine_step */
    double h2d_bytes, d2h_bytes;     /* traffic, total */
    unsigned long errors;            /* device errors (each one also logs) */
} asr_engine_stats;

typedef struct {
    void (*close)(asr_engine *e);
    void (*facts)(const asr_engine *e, asr_engine_facts *f);
    void (*stats)(const asr_engine *e, asr_engine_stats *s);
    int (*lang_id)(const asr_engine *e, const char *lang);
    int (*lookahead_ok)(const asr_engine *e, int lookahead);
    int (*slot_reset)(asr_engine *e, int slot, const char *lang, int lookahead);
    size_t (*slot_need_samples)(const asr_engine *e, int slot);
    int (*slot_feed)(asr_engine *e, int slot, const float *pcm, size_t n);
    int (*slot_ready)(const asr_engine *e, int slot);
    double (*slot_audio_s)(const asr_engine *e, int slot);
    const char *(*slot_text)(const asr_engine *e, int slot);
    const char *(*slot_lang)(const asr_engine *e, int slot);
    int (*step)(asr_engine *e, const asr_step_req *reqs, int n, asr_step_out *outs);
    int (*dead)(const asr_engine *e);
    const char *(*error)(const asr_engine *e);
    size_t (*dispatch_map)(const asr_engine *e, char *buf, size_t cap);
} asr_engine_ops;

struct asr_engine { const asr_engine_ops *ops; };

/* Open: loads the pack, uploads the weights, allocates the arena for cfg->cap
 * slots. NULL on failure with the reason in `err`. */
asr_engine *asr_engine_open_cuda(const asr_engine_cfg *cfg, char *err, size_t errcap);
asr_engine *asr_engine_open_cpu(const asr_engine_cfg *cfg, char *err, size_t errcap);

static inline void asr_engine_close(asr_engine *e) { if (e) e->ops->close(e); }
static inline void asr_engine_get_facts(const asr_engine *e, asr_engine_facts *f) { e->ops->facts(e, f); }
static inline void asr_engine_get_stats(const asr_engine *e, asr_engine_stats *s) { e->ops->stats(e, s); }
/* Language tag -> prompt id (-1 = not served); NULL/"auto" -> the default. A
 * config lookup, never inference. */
static inline int asr_engine_lang_id(const asr_engine *e, const char *lang) { return e->ops->lang_id(e, lang); }
static inline int asr_engine_lookahead_ok(const asr_engine *e, int la) { return e->ops->lookahead_ok(e, la); }
/* A new utterance on slot `slot`: caches emptied, decoder at SOS, mel reset,
 * text emptied; `lang` is the tag (NULL/"auto" = the pack's default). 0 = ok. */
static inline int asr_engine_slot_reset(asr_engine *e, int slot, const char *lang, int la) { return e->ops->slot_reset(e, slot, lang, la); }
/* Samples the slot still needs before its next chunk is complete. */
static inline size_t asr_engine_slot_need_samples(const asr_engine *e, int slot) { return e->ops->slot_need_samples(e, slot); }
/* Samples of the slot's utterance (16 kHz mono f32), engine thread only. */
static inline int asr_engine_slot_feed(asr_engine *e, int slot, const float *pcm, size_t n) { return e->ops->slot_feed(e, slot, pcm, n); }
static inline int asr_engine_slot_ready(const asr_engine *e, int slot) { return e->ops->slot_ready(e, slot); }
static inline double asr_engine_slot_audio_s(const asr_engine *e, int slot) { return e->ops->slot_audio_s(e, slot); }
/* The whole transcript of the slot's current utterance (engine-owned). */
static inline const char *asr_engine_slot_text(const asr_engine *e, int slot) { return e->ops->slot_text(e, slot); }
static inline const char *asr_engine_slot_lang(const asr_engine *e, int slot) { return e->ops->slot_lang(e, slot); }
/* ONE cohort step: every requested slot advances by at most one chunk (or its
 * tail). outs[i] answers reqs[i]. Returns 0, or -1 when the DEVICE failed --
 * then the engine is dead and the caller ends every session visibly. */
static inline int asr_engine_step(asr_engine *e, const asr_step_req *r, int n, asr_step_out *o) { return e->ops->step(e, r, n, o); }
static inline int asr_engine_dead(const asr_engine *e) { return e->ops->dead(e); }
static inline const char *asr_engine_error(const asr_engine *e) { return e->ops->error(e); }
/* The kernel that runs for each operation, one line per op, as the process
 * resolved it (ENGINEERING.md §5). Writes into `buf`, returns the bytes. */
static inline size_t asr_engine_dispatch_map(const asr_engine *e, char *buf, size_t cap) { return e->ops->dispatch_map(e, buf, cap); }

#endif
