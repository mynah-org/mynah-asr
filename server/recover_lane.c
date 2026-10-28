/* Recovery lanes. See recover_lane.h for the contract. */
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#include "recover_lane.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/syscall.h>
#endif
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

#include "../src/threads.h"
#include "http_util.h"
#include "slot.h"

#define RLANE_MAX 16
#define RLANE_TEXT_CAP 512
#define RLANE_PRESETS 8

enum { RL_IDLE = 0, RL_BUSY, RL_DONE };

typedef struct {
    pthread_t th;
    int started;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int state;

    /* recoverers, one per lookahead preset, opened once */
    int la[RLANE_PRESETS];
    mynah_asr_recoverer *r[RLANE_PRESETS];
    int n_r;

    /* the job (lane-owned while BUSY) */
    int slot_id;
    unsigned gen;
    int lookahead;
    char lang[MYNAH_ASR_SLOT_LANG_CAP];
    float *cur, *prev;
    size_t n_cur, n_prev;
    char prev_text[RLANE_TEXT_CAP];
    int has_prev_text;

    /* the result (scheduler-read while DONE) */
    mynah_asr_rlane_result res;
    mynah_asr_recoverer *used;
} rlane;

static struct {
    mynah_asr_model *m;
    mynah_asr_recover_cfg cfg;
    rlane lanes[RLANE_MAX];
    int n;
    size_t cur_cap, prev_cap;
    void (*wake)(void);
    _Atomic int stop;
    _Atomic int done;     /* jobs in DONE: lets collect cost one load when idle */
} G;

/* Below the scheduler and the HTTP threads. Linux: a per-thread nice (the tid
 * is the scheduling entity; setpriority on it moves this thread only). macOS:
 * the utility QoS class, the per-thread knob the platform offers. Best effort:
 * a refusal (no permission) leaves the default priority and says so once. */
static void rlane_lower_priority(void) {
#if defined(__linux__)
    const pid_t tid = (pid_t)syscall(SYS_gettid);
    if (setpriority(PRIO_PROCESS, (id_t)tid, 10) != 0)
        fprintf(stderr, "mynah-asr-server: the recovery lane could not lower its "
                        "priority; it runs at the default one\n");
#elif defined(__APPLE__)
    if (pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0) != 0)
        fprintf(stderr, "mynah-asr-server: the recovery lane could not lower its "
                        "QoS class; it runs at the default one\n");
#endif
}

static mynah_asr_recoverer *rlane_open(rlane *l, int lookahead) {
    for (int i = 0; i < l->n_r; i++)
        if (l->la[i] == lookahead) return l->r[i];
    if (l->n_r >= RLANE_PRESETS) return NULL;
    mynah_asr_recoverer *r = mynah_asr_recoverer_open(G.m, lookahead, &G.cfg);
    if (r == NULL) return NULL;
    l->la[l->n_r] = lookahead;
    l->r[l->n_r] = r;
    l->n_r++;
    return r;
}

static void *rlane_main(void *arg) {
    rlane *l = (rlane *)arg;
    mynah_asr_thread_set_name("mynah-recover");
    mynah_asr_thread_set_serial(1);
    rlane_lower_priority();
    for (;;) {
        pthread_mutex_lock(&l->mu);
        while (l->state != RL_BUSY && !atomic_load_explicit(&G.stop, memory_order_acquire))
            pthread_cond_wait(&l->cv, &l->mu);
        if (l->state != RL_BUSY) { pthread_mutex_unlock(&l->mu); break; }
        pthread_mutex_unlock(&l->mu);

        const double t0 = mynah_asr_now();
        mynah_asr_recover_result rr;
        memset(&rr, 0, sizeof(rr));
        mynah_asr_recoverer *r = rlane_open(l, l->lookahead);
        int rc = -1;
        if (r != NULL)
            rc = mynah_asr_recover_turn(r, l->lang, l->cur, l->n_cur,
                                        l->n_prev > 0 ? l->prev : NULL, l->n_prev,
                                        l->has_prev_text ? l->prev_text : NULL, &rr);

        pthread_mutex_lock(&l->mu);
        l->used = r;
        l->res.slot_id = l->slot_id;
        l->res.gen = l->gen;
        l->res.rc = rc;
        l->res.accepted = rc == 0 && rr.accepted;
        l->res.kind = rr.winner_kind;
        snprintf(l->res.lang, sizeof(l->res.lang), "%s", rr.lang);
        l->res.text = NULL;
        l->res.seconds = mynah_asr_now() - t0;
        l->state = RL_DONE;
        pthread_mutex_unlock(&l->mu);
        atomic_fetch_add_explicit(&G.done, 1, memory_order_release);
        if (G.wake) G.wake();
    }
    return NULL;
}

int mynah_asr_rlanes_start(mynah_asr_model *m, int lanes, int lookahead,
                           const mynah_asr_recover_cfg *cfg, void (*wake)(void)) {
    memset(&G, 0, sizeof(G));
    if (lanes < 1) lanes = 1;
    if (lanes > RLANE_MAX) lanes = RLANE_MAX;
    G.m = m;
    G.cfg = *cfg;
    G.wake = wake;
    const int sr = mynah_asr_sample_rate(m) > 0 ? mynah_asr_sample_rate(m) : 16000;
    G.cur_cap = (size_t)(cfg->max_turn_s * (double)sr);
    G.prev_cap = (size_t)(cfg->replay_frames > 0 ? cfg->replay_frames : 0) *
                 (size_t)mynah_asr_frame_samples(m);
    for (int i = 0; i < lanes; i++) {
        rlane *l = &G.lanes[i];
        if (pthread_mutex_init(&l->mu, NULL) != 0 || pthread_cond_init(&l->cv, NULL) != 0)
            goto fail;
        G.n = i + 1;
        l->cur = (float *)malloc((G.cur_cap > 0 ? G.cur_cap : 1) * sizeof(float));
        l->prev = (float *)malloc((G.prev_cap > 0 ? G.prev_cap : 1) * sizeof(float));
        if (!l->cur || !l->prev) goto fail;
        /* The session default preset, opened here: the steady state allocates
         * nothing (the detokenised attempt texts are recover.c's declared
         * exception). */
        if (rlane_open(l, lookahead) == NULL) goto fail;
    }
    for (int i = 0; i < G.n; i++) {
        if (pthread_create(&G.lanes[i].th, NULL, rlane_main, &G.lanes[i]) != 0) goto fail;
        G.lanes[i].started = 1;
    }
    return 0;
fail:
    mynah_asr_rlanes_stop();
    return -1;
}

void mynah_asr_rlanes_stop(void) {
    atomic_store_explicit(&G.stop, 1, memory_order_release);
    for (int i = 0; i < G.n; i++) {
        rlane *l = &G.lanes[i];
        pthread_mutex_lock(&l->mu);
        pthread_cond_broadcast(&l->cv);
        pthread_mutex_unlock(&l->mu);
    }
    for (int i = 0; i < G.n; i++) {
        rlane *l = &G.lanes[i];
        if (l->started) pthread_join(l->th, NULL);
        l->started = 0;
        for (int k = 0; k < l->n_r; k++) mynah_asr_recoverer_close(l->r[k]);
        l->n_r = 0;
        free(l->cur);
        free(l->prev);
        l->cur = l->prev = NULL;
        pthread_cond_destroy(&l->cv);
        pthread_mutex_destroy(&l->mu);
    }
    G.n = 0;
}

int mynah_asr_rlanes_submit(int slot_id, unsigned gen, int lookahead, const char *lang,
                            const float *cur, size_t n_cur, const float *prev,
                            size_t n_prev, const char *prev_text) {
    if (n_cur == 0 || n_cur > G.cur_cap) return -1;
    if (n_prev > G.prev_cap) {               /* keep the tail: that is the replay */
        prev += n_prev - G.prev_cap;
        n_prev = G.prev_cap;
    }
    for (int i = 0; i < G.n; i++) {
        rlane *l = &G.lanes[i];
        pthread_mutex_lock(&l->mu);
        if (l->state != RL_IDLE) { pthread_mutex_unlock(&l->mu); continue; }
        l->slot_id = slot_id;
        l->gen = gen;
        l->lookahead = lookahead;
        snprintf(l->lang, sizeof(l->lang), "%s", lang ? lang : "auto");
        memcpy(l->cur, cur, n_cur * sizeof(float));
        l->n_cur = n_cur;
        if (prev != NULL && n_prev > 0) memcpy(l->prev, prev, n_prev * sizeof(float));
        l->n_prev = prev != NULL ? n_prev : 0;
        l->has_prev_text = prev_text != NULL;
        snprintf(l->prev_text, sizeof(l->prev_text), "%s", prev_text ? prev_text : "");
        l->state = RL_BUSY;
        pthread_cond_signal(&l->cv);
        pthread_mutex_unlock(&l->mu);
        return 0;
    }
    return -1;
}

int mynah_asr_rlanes_collect(void (*fn)(const mynah_asr_rlane_result *r, void *ud),
                             void *ud) {
    int n = 0;
    if (atomic_load_explicit(&G.done, memory_order_acquire) == 0) return 0;
    for (int i = 0; i < G.n; i++) {
        rlane *l = &G.lanes[i];
        pthread_mutex_lock(&l->mu);
        const int done = l->state == RL_DONE;
        pthread_mutex_unlock(&l->mu);
        if (!done) continue;
        /* DONE: the lane waits for BUSY and touches nothing, so the result and
         * the recoverer's text are read without the lock. */
        mynah_asr_rlane_result r = l->res;
        r.text = (r.accepted && l->used != NULL) ? mynah_asr_recover_text(l->used) : "";
        fn(&r, ud);
        pthread_mutex_lock(&l->mu);
        l->state = RL_IDLE;
        pthread_mutex_unlock(&l->mu);
        atomic_fetch_sub_explicit(&G.done, 1, memory_order_relaxed);
        n++;
    }
    return n;
}
