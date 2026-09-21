/* The slot table's element. See slot.h for the ownership contract. */
#include "slot.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* See mynah_asr_slot_set_notify: one scheduler per process, so one hook. */
static void (*g_notify)(void);

/* Process-wide count of slots holding a whole chunk. Maintained by the two
 * paths that change `len`, each under the slot's own mutex. */
static _Atomic int g_ready_slots;

int mynah_asr_slot_ready_count(void) {
    const int n = atomic_load_explicit(&g_ready_slots, memory_order_relaxed);
    return n > 0 ? n : 0;
}

static unsigned long slot_self(void) {
    return (unsigned long)(uintptr_t)pthread_self();
}
static void slot_lock(mynah_asr_slot *s, const char *where) {
    pthread_mutex_lock(&s->mu);
    atomic_store_explicit(&s->mu_owner, slot_self(), memory_order_relaxed);
    atomic_store_explicit(&s->mu_since, mynah_asr_now(), memory_order_relaxed);
    atomic_store_explicit(&s->mu_where, where, memory_order_relaxed);
}
static void slot_unlock(mynah_asr_slot *s) {
    atomic_store_explicit(&s->mu_owner, 0ul, memory_order_relaxed);
    pthread_mutex_unlock(&s->mu);
}
/* A cond_wait releases the mutex while it waits, so the trace must say so or a
 * dump would name a thread that is not holding anything -- and the thread most
 * likely to be named wrongly is the one waiting in mynah_asr_slot_wait_done at
 * the end of a stream, which is exactly the moment the stall under
 * investigation occurs. Both waits are wrapped for that reason. */
static void slot_cond_wait(mynah_asr_slot *s, pthread_cond_t *cv, const char *where) {
    atomic_store_explicit(&s->mu_owner, 0ul, memory_order_relaxed);
    pthread_cond_wait(cv, &s->mu);
    atomic_store_explicit(&s->mu_owner, slot_self(), memory_order_relaxed);
    atomic_store_explicit(&s->mu_since, mynah_asr_now(), memory_order_relaxed);
    atomic_store_explicit(&s->mu_where, where, memory_order_relaxed);
}

static int slot_cond_timedwait(mynah_asr_slot *s, pthread_cond_t *cv,
                               const struct timespec *dl, const char *where) {
    atomic_store_explicit(&s->mu_owner, 0ul, memory_order_relaxed);
    const int rc = pthread_cond_timedwait(cv, &s->mu, dl);
    /* The owner is restored on EVERY return, timeout included: a timed wait
     * that expires still comes back holding the mutex. */
    atomic_store_explicit(&s->mu_owner, slot_self(), memory_order_relaxed);
    atomic_store_explicit(&s->mu_since, mynah_asr_now(), memory_order_relaxed);
    atomic_store_explicit(&s->mu_where, where, memory_order_relaxed);
    return rc;
}

/* Set or clear this slot's readiness and keep the global count in step. Called
 * with the slot's mutex held, by every path that changes `len` or `need`. */
static void slot_refresh_ready_locked(mynah_asr_slot *s) {
    const size_t need = atomic_load_explicit(&s->need_hint, memory_order_relaxed);
    const int want = need > 0 && s->len >= need;
    const int had = atomic_load_explicit(&s->ready_flag, memory_order_relaxed);
    if (want == had) return;
    atomic_store_explicit(&s->ready_flag, want, memory_order_relaxed);
    /* The rising edge only. This function returns early when the flag does not
     * change, so a slot that stays ready ACROSS a take is not re-stamped here
     * -- the take path does that itself, for the reason written there. */
    if (want)
        atomic_store_explicit(&s->ready_since, mynah_asr_now(), memory_order_relaxed);
    atomic_fetch_add_explicit(&g_ready_slots, want ? 1 : -1, memory_order_relaxed);
}

int mynah_asr_slot_is_ready(const mynah_asr_slot *s) {
    return atomic_load_explicit(&s->ready_flag, memory_order_relaxed);
}

double mynah_asr_slot_ready_since(const mynah_asr_slot *s) {
    return atomic_load_explicit(&s->ready_since, memory_order_relaxed);
}

void mynah_asr_slot_set_need(mynah_asr_slot *s, size_t need_samples) {
    slot_lock(s, "mynah_asr_slot_set_need");
    atomic_store_explicit(&s->need_hint, need_samples, memory_order_relaxed);
    slot_refresh_ready_locked(s);
    slot_unlock(s);
}


/* Take/release a slot's mutex and leave a trace of who did it. `where` is a
 * literal, so storing the pointer costs nothing and a dump can print it. */


void mynah_asr_slot_set_notify(void (*notify)(void)) { g_notify = notify; }

double mynah_asr_now(void) {
    struct timespec ts;
#if defined(CLOCK_MONOTONIC)
    clock_gettime(CLOCK_MONOTONIC, &ts);
#else
    clock_gettime(CLOCK_REALTIME, &ts);
#endif
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

const char *mynah_asr_slot_cancel_code(mynah_asr_slot_cancel r) {
    switch (r) {
        case MYNAH_ASR_SLOT_CANCEL_IDLE:     return "idle_timeout";
        case MYNAH_ASR_SLOT_CANCEL_FRAME:    return "frame_too_large";
        case MYNAH_ASR_SLOT_CANCEL_PROTOCOL: return "protocol_error";
        case MYNAH_ASR_SLOT_CANCEL_PEER:     return "peer_gone";
        case MYNAH_ASR_SLOT_CANCEL_SHUTDOWN: return "shutting_down";
        case MYNAH_ASR_SLOT_CANCEL_NONE:     break;
    }
    return "cancelled";
}

int mynah_asr_slot_init(mynah_asr_slot *s, int id, size_t ring_samples,
                        size_t take_samples) {
    memset(s, 0, sizeof(*s));
    s->id = id;
    s->stream_lookahead = -0x7fffffff;   /* no pooled stream yet */
    if (pthread_mutex_init(&s->mu, NULL) != 0) return -1;
    if (pthread_cond_init(&s->space, NULL) != 0) {
        pthread_mutex_destroy(&s->mu);
        return -1;
    }
    s->cap = ring_samples;
    s->ring = (float *)malloc(s->cap * sizeof(float));
    /* One record per push, so the resolution of the lag attribution is the
     * client's frame size. 50 per second of ring is 20 ms at the finest, well
     * under any sane WebSocket frame; a fuller ring merges the two newest
     * arrivals rather than dropping the oldest, which would mis-date the audio
     * the scheduler is about to consume. */
    s->arr_cap = ring_samples / 320 + 64;
    s->arr = (mynah_asr_slot_arrival *)malloc(s->arr_cap * sizeof(*s->arr));
    s->take_cap = take_samples;
    s->take = (float *)malloc(s->take_cap * sizeof(float));
    s->frame_cap = 32u * 1024u;
    s->frame = (unsigned char *)malloc(s->frame_cap);
    if (!s->ring || !s->arr || !s->take || !s->frame) return -1;
    return 0;
}

void mynah_asr_slot_destroy(mynah_asr_slot *s) {
    if (s->stream) mynah_asr_stream_close(s->stream);
    s->stream = NULL;
    free(s->ring);
    free(s->arr);
    free(s->take);
    free(s->frame);
    s->ring = NULL; s->arr = NULL; s->take = NULL; s->frame = NULL;
    pthread_cond_destroy(&s->space);
    pthread_mutex_destroy(&s->mu);
}

/* Everything a new session must not inherit from the previous one. The stream
 * object and its lookahead are deliberately NOT here: they are the pool. */
static void slot_reset_queue_locked(mynah_asr_slot *s) {
    s->head = s->len = 0;
    s->arr_head = s->arr_len = 0;
    s->samples_in = s->samples_consumed = 0;
    s->req = 0;
    s->cancel_reason = MYNAH_ASR_SLOT_CANCEL_NONE;
    s->req_lang[0] = '\0';
}

int mynah_asr_slot_claim(mynah_asr_slot *s, const char *lang, int lookahead,
                         mynah_asr_stream_out *out, double now) {
    slot_lock(s, "mynah_asr_slot_claim");
    if (s->state != MYNAH_ASR_SLOT_FREE) {
        slot_unlock(s);
        return -1;
    }
    slot_reset_queue_locked(s);
    s->state = MYNAH_ASR_SLOT_ACTIVE;
    slot_unlock(s);

    /* Scheduler-side fields: written before the slot can be seen as ACTIVE by a
     * step, because the store above is the publication point. */
    s->out = out;
    s->lookahead = lookahead;
    snprintf(s->lang, sizeof(s->lang), "%s", lang ? lang : "auto");
    s->seq = 0;
    s->needs_reset = 1;   /* a pooled stream always starts a session with a reset */
    s->t_open = now;
    s->t_first_delta = 0.0;
    s->t_first_audio = s->t_first_queued = 0.0;
    s->last_arrival = 0.0;
    s->t_last_step = 0.0;   /* a pooled slot must not inherit the last session's */
    s->lag_sum_ms = s->lag_max_ms = 0.0;
    s->deltas = s->eous = s->steps = 0;
    memset(s->lag_hist, 0, sizeof(s->lag_hist));
    return 0;
}

void mynah_asr_slot_arm(mynah_asr_slot *s, mynah_asr_stream_out *out) {
    slot_lock(s, "mynah_asr_slot_arm");
    s->out = out;
    slot_unlock(s);
    if (g_notify) g_notify();
}

void mynah_asr_slot_release(mynah_asr_slot *s) {
    slot_lock(s, "mynah_asr_slot_release");
    atomic_store_explicit(&s->need_hint, (size_t)0, memory_order_relaxed);
    slot_refresh_ready_locked(s);
    slot_reset_queue_locked(s);
    s->state = MYNAH_ASR_SLOT_FREE;
    s->out = NULL;
    pthread_cond_broadcast(&s->space);
    slot_unlock(s);
}

mynah_asr_slot_state mynah_asr_slot_poll(mynah_asr_slot *s,
                                         mynah_asr_stream_out **out, size_t *avail) {
    slot_lock(s, "mynah_asr_slot_poll");
    const mynah_asr_slot_state st = s->state;
    if (out) *out = s->out;
    if (avail) *avail = s->len;
    slot_unlock(s);
    return st;
}

mynah_asr_slot_state mynah_asr_slot_get_state(mynah_asr_slot *s) {
    slot_lock(s, "mynah_asr_slot_get_state");
    const mynah_asr_slot_state st = s->state;
    slot_unlock(s);
    return st;
}

void mynah_asr_slot_set_state(mynah_asr_slot *s, mynah_asr_slot_state st) {
    slot_lock(s, "mynah_asr_slot_set_state");
    s->state = st;
    /* A pusher parked on a full ring must not outlive the session. */
    if (st == MYNAH_ASR_SLOT_DONE || st == MYNAH_ASR_SLOT_FREE)
        pthread_cond_broadcast(&s->space);
    slot_unlock(s);
}

/* True while the pusher still has a session to push into. */
static int slot_push_open_locked(const mynah_asr_slot *s) {
    return s->state != MYNAH_ASR_SLOT_FREE && s->state != MYNAH_ASR_SLOT_DONE &&
           (s->req & MYNAH_ASR_SLOT_REQ_CANCEL) == 0;
}

static void slot_record_arrival_locked(mynah_asr_slot *s, double now) {
    if (s->t_first_audio == 0.0) s->t_first_audio = now;
    if (s->arr_len == s->arr_cap) {
        /* Merge into the newest record instead of evicting the oldest: the
         * oldest is what dates the audio the scheduler consumes next. */
        const size_t last = (s->arr_head + s->arr_len - 1) % s->arr_cap;
        s->arr[last].end = s->samples_in;
        s->arr[last].t = now;
        return;
    }
    const size_t tail = (s->arr_head + s->arr_len) % s->arr_cap;
    s->arr[tail].end = s->samples_in;
    s->arr[tail].t = now;
    s->arr_len++;
}

size_t mynah_asr_slot_push(mynah_asr_slot *s, const float *samples, size_t n,
                           double now) {
    size_t done = 0;
    slot_lock(s, "mynah_asr_slot_push");
    while (done < n) {
        while (s->len == s->cap && slot_push_open_locked(s))
            slot_cond_wait(s, &s->space, "mynah_asr_slot_push");
        if (!slot_push_open_locked(s)) break;
        size_t room = s->cap - s->len;
        size_t take = n - done;
        if (take > room) take = room;
        const size_t tail = (s->head + s->len) % s->cap;
        const size_t first = take < s->cap - tail ? take : s->cap - tail;
        memcpy(s->ring + tail, samples + done, first * sizeof(float));
        if (first < take)
            memcpy(s->ring, samples + done + first, (take - first) * sizeof(float));
        s->len += take;
        s->samples_in += take;
        done += take;
        slot_record_arrival_locked(s, now);
        slot_refresh_ready_locked(s);
        /* Ring the doorbell from inside the lock, and before parking again on a
         * full ring: a push that fills the ring must wake the scheduler, or the
         * two would end up waiting for each other. */
        if (g_notify) g_notify();
    }
    slot_unlock(s);
    return done;
}

size_t mynah_asr_slot_take(mynah_asr_slot *s, float *dst, size_t n, double *arrival) {
    slot_lock(s, "mynah_asr_slot_take");
    if (n > s->len) n = s->len;
    const size_t first = n < s->cap - s->head ? n : s->cap - s->head;
    memcpy(dst, s->ring + s->head, first * sizeof(float));
    if (first < n) memcpy(dst + first, s->ring, (n - first) * sizeof(float));
    s->head = (s->head + n) % s->cap;
    s->len -= n;
    s->samples_consumed += n;

    /* Date the last sample taken: drop every arrival wholly behind it, then read
     * the one that carried it. An empty record set (only possible after a merge)
     * falls back to the newest time we know. */
    double t = 0.0;
    while (s->arr_len > 0 && s->arr[s->arr_head].end < s->samples_consumed) {
        s->arr_head = (s->arr_head + 1) % s->arr_cap;
        s->arr_len--;
    }
    if (s->arr_len > 0) t = s->arr[s->arr_head].t;
    if (arrival) *arrival = t;

    pthread_cond_broadcast(&s->space);
    slot_refresh_ready_locked(s);
    /* Still ready after the drain: the NEXT chunk is executable from this
     * instant and has waited nothing yet, so its clock starts now. Without
     * this the stamp keeps the moment the slot FIRST went ready and never
     * moves while the slot stays behind, so ready->selected would report the
     * backlog instead of the delay the scheduler added -- the backlog counted
     * a second time, under a name that suggests a scheduler defect. Caught by
     * a 3-stream smoke that read a 2.5 s p95 where the true value is near
     * zero; it would have read as a scheduling stall on a nearly idle box. */
    if (atomic_load_explicit(&s->ready_flag, memory_order_relaxed))
        atomic_store_explicit(&s->ready_since, mynah_asr_now(),
                              memory_order_relaxed);
    slot_unlock(s);
    return n;
}

size_t mynah_asr_slot_available(mynah_asr_slot *s) {
    slot_lock(s, "mynah_asr_slot_available");
    const size_t n = s->len;
    slot_unlock(s);
    return n;
}

void mynah_asr_slot_request(mynah_asr_slot *s, int flags, const char *lang,
                            mynah_asr_slot_cancel reason) {
    slot_lock(s, "mynah_asr_slot_request");
    s->req |= flags;
    if ((flags & MYNAH_ASR_SLOT_REQ_RESET) != 0)
        snprintf(s->req_lang, sizeof(s->req_lang), "%s", lang ? lang : "");
    if ((flags & MYNAH_ASR_SLOT_REQ_CANCEL) != 0 &&
        s->cancel_reason == MYNAH_ASR_SLOT_CANCEL_NONE)
        s->cancel_reason = reason;
    if ((flags & MYNAH_ASR_SLOT_REQ_CANCEL) != 0)
        pthread_cond_broadcast(&s->space);
    slot_unlock(s);
    if (g_notify) g_notify();
}

int mynah_asr_slot_wait_done(mynah_asr_slot *s, int timeout_ms) {
    struct timespec dl;
    clock_gettime(CLOCK_REALTIME, &dl);
    dl.tv_sec += timeout_ms / 1000;
    dl.tv_nsec += (long)(timeout_ms % 1000) * 1000000L;
    if (dl.tv_nsec >= 1000000000L) { dl.tv_sec++; dl.tv_nsec -= 1000000000L; }

    slot_lock(s, "mynah_asr_slot_wait_done");
    while (s->state != MYNAH_ASR_SLOT_DONE && s->state != MYNAH_ASR_SLOT_FREE) {
        if (slot_cond_timedwait(s, &s->space, &dl, "mynah_asr_slot_wait_done") != 0)
            break;
    }
    const int done = s->state == MYNAH_ASR_SLOT_DONE || s->state == MYNAH_ASR_SLOT_FREE;
    slot_unlock(s);
    return done;
}

int mynah_asr_slot_take_requests(mynah_asr_slot *s, char *lang,
                                 mynah_asr_slot_cancel *reason) {
    slot_lock(s, "mynah_asr_slot_take_requests");
    const int req = s->req;
    if (lang) snprintf(lang, MYNAH_ASR_SLOT_LANG_CAP, "%s", s->req_lang);
    if (reason) *reason = s->cancel_reason;
    /* CANCEL survives: it is terminal, and a later poll must still see it. */
    s->req &= MYNAH_ASR_SLOT_REQ_CANCEL;
    slot_unlock(s);
    return req;
}
