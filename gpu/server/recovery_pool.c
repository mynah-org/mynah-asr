/* Empty-turn recovery on spare engine slots (recovery_pool.h). */
#include "recovery_pool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int slot;                    /* engine slot */
    mynah_asr_attempt a;
    float *buf;                  /* the spare slot's buffer: the attempt's samples */
    size_t len, off;             /* samples built / fed */
    int done;                    /* the padded tail ran */
} rp_attempt;

typedef struct {
    int used;
    int owner;
    unsigned gen;
    unsigned long seq;           /* start order: oldest job first */
    char lang[16];
    int lag;                     /* TEXT's window: cfg.lag_frames or lookahead + 1 */
    int n;
    rp_attempt att[MYNAH_ASR_RECOVER_MAX_ATTEMPTS];
    char *prev_text;
    double t0;
} rp_job;

struct rpool {
    asr_engine *e;
    const mynah_asr_tokenizer *tk;
    rpool_cfg c;
    size_t max_turn, buf_cap;
    float **bufs;                /* one per spare slot */
    int *slot_job;               /* spare slot -> job (-1 free) */
    int *slot_att;               /* spare slot -> attempt index */
    rp_job *jobs;                /* n_slots entries: a job holds at least one slot */
    unsigned long seq;
    char *texts[MYNAH_ASR_RECOVER_MAX_ATTEMPTS];
    rpool_stats st;
};

rpool *rpool_open(asr_engine *e, const mynah_asr_tokenizer *tk, const rpool_cfg *cfg) {
    if (!e || !tk || !cfg || cfg->n_slots < 1 || cfg->frame_samples < 1 || cfg->sample_rate < 1) return NULL;
    rpool *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->e = e; p->tk = tk; p->c = *cfg;
    int longest = cfg->cfg.replay_frames;
    for (int i = 0; i < cfg->cfg.n_phases; i++)
        if (cfg->cfg.phase_frames[i] > longest) longest = cfg->cfg.phase_frames[i];
    p->max_turn = (size_t)(cfg->cfg.max_turn_s * (double)cfg->sample_rate);
    p->buf_cap = (size_t)longest * (size_t)cfg->frame_samples + p->max_turn;
    const int n = cfg->n_slots;
    p->bufs = calloc((size_t)n, sizeof(float *));
    p->slot_job = malloc((size_t)n * sizeof(int));
    p->slot_att = calloc((size_t)n, sizeof(int));
    p->jobs = calloc((size_t)n, sizeof(rp_job));
    if (!p->bufs || !p->slot_job || !p->slot_att || !p->jobs) { rpool_close(p); return NULL; }
    for (int i = 0; i < n; i++) {
        p->slot_job[i] = -1;
        if (!(p->bufs[i] = malloc((p->buf_cap ? p->buf_cap : 1) * sizeof(float)))) { rpool_close(p); return NULL; }
    }
    return p;
}

void rpool_close(rpool *p) {
    if (!p) return;
    if (p->bufs) for (int i = 0; i < p->c.n_slots; i++) free(p->bufs[i]);
    if (p->jobs) for (int i = 0; i < p->c.n_slots; i++) free(p->jobs[i].prev_text);
    for (int i = 0; i < MYNAH_ASR_RECOVER_MAX_ATTEMPTS; i++) free(p->texts[i]);
    free(p->bufs); free(p->slot_job); free(p->slot_att); free(p->jobs);
    free(p);
}

size_t rpool_max_turn_samples(const rpool *p) { return p->max_turn; }

static void job_free(rpool *p, int j) {
    rp_job *jb = &p->jobs[j];
    for (int a = 0; a < jb->n; a++) p->slot_job[jb->att[a].slot - p->c.first_slot] = -1;
    free(jb->prev_text);
    memset(jb, 0, sizeof(*jb));
}

int rpool_start(rpool *p, int owner, unsigned gen, const char *lang, int lookahead,
                const float *cur, size_t n_cur, const float *prev, size_t n_prev,
                const char *prev_text, double now) {
    if (!cur || n_cur == 0 || n_cur > p->max_turn) { p->st.out_of_bounds++; return RP_START_BOUNDS; }
    mynah_asr_attempt plan[MYNAH_ASR_RECOVER_MAX_ATTEMPTS];
    const int n = mynah_asr_recover_plan(&p->c.cfg, prev ? n_prev : 0, p->c.frame_samples, plan);
    if (n <= 0) { p->st.out_of_bounds++; return RP_START_BOUNDS; }
    /* every attempt on a free spare slot, or nothing */
    int slots[MYNAH_ASR_RECOVER_MAX_ATTEMPTS], got = 0, j = -1;
    for (int i = 0; i < p->c.n_slots && got < n; i++)
        if (p->slot_job[i] < 0) slots[got++] = i;
    for (int i = 0; i < p->c.n_slots && j < 0; i++)
        if (!p->jobs[i].used) j = i;
    if (got < n || j < 0) { p->st.busy++; return RP_START_BUSY; }
    rp_job *jb = &p->jobs[j];
    memset(jb, 0, sizeof(*jb));
    for (int a = 0; a < n; a++) {
        const int es = p->c.first_slot + slots[a];
        if (asr_engine_slot_reset(p->e, es, lang, lookahead) != 0) return RP_START_ERROR;
    }
    jb->used = 1;
    jb->owner = owner;
    jb->gen = gen;
    jb->seq = ++p->seq;
    snprintf(jb->lang, sizeof(jb->lang), "%s", lang ? lang : "auto");
    jb->lag = p->c.cfg.lag_frames > 0 ? p->c.cfg.lag_frames : lookahead + 1;
    jb->n = n;
    jb->t0 = now;
    jb->prev_text = prev_text ? strdup(prev_text) : NULL;
    for (int a = 0; a < n; a++) {
        rp_attempt *at = &jb->att[a];
        at->slot = p->c.first_slot + slots[a];
        at->a = plan[a];
        at->buf = p->bufs[slots[a]];
        at->len = mynah_asr_recover_build(&plan[a], p->c.frame_samples, cur, n_cur, prev, n_prev, at->buf);
        at->off = 0;
        at->done = 0;
        p->slot_job[slots[a]] = j;
        p->slot_att[slots[a]] = a;
    }
    p->st.started++;
    return j;
}

void rpool_cancel(rpool *p, int job) {
    if (job < 0 || job >= p->c.n_slots || !p->jobs[job].used) return;
    job_free(p, job);
    p->st.cancelled++;
}

/* jobs in start order */
static int job_order(const rpool *p, int *order) {
    int n = 0;
    for (int j = 0; j < p->c.n_slots; j++)
        if (p->jobs[j].used) {
            int k = n++;
            while (k > 0 && p->jobs[order[k - 1]].seq > p->jobs[j].seq) { order[k] = order[k - 1]; k--; }
            order[k] = j;
        }
    return n;
}

int rpool_lanes(rpool *p, asr_step_req *reqs, int room) {
    int order[RPOOL_MAX_SLOTS], k = 0;
    if (room <= 0 || p->c.n_slots > RPOOL_MAX_SLOTS) return 0;
    const int nj = job_order(p, order);
    for (int o = 0; o < nj && k < room; o++) {
        rp_job *jb = &p->jobs[order[o]];
        for (int a = 0; a < jb->n && k < room; a++) {
            rp_attempt *at = &jb->att[a];
            if (at->done) continue;
            const size_t need = asr_engine_slot_need_samples(p->e, at->slot);
            if (at->off < at->len && need > 0) {
                size_t take = need;
                if (at->off + take > at->len) take = at->len - at->off;
                if (asr_engine_slot_feed(p->e, at->slot, at->buf + at->off, take) != 0) return -1;
                at->off += take;
            }
            const int fin = at->off >= at->len;
            if (asr_engine_slot_ready(p->e, at->slot) || fin) {
                reqs[k].slot = at->slot;
                reqs[k].finalize = fin;
                k++;
            }
        }
    }
    return k;
}

void rpool_stepped(rpool *p, const asr_step_req *reqs, const asr_step_out *outs, int k) {
    p->st.lanes += (unsigned long)(k > 0 ? k : 0);
    for (int i = 0; i < k; i++) {
        const int s = reqs[i].slot - p->c.first_slot;
        if (s < 0 || s >= p->c.n_slots || p->slot_job[s] < 0 || !outs[i].finished) continue;
        rp_job *jb = &p->jobs[p->slot_job[s]];
        rp_attempt *at = &jb->att[p->slot_att[s]];
        if (!at->done) { at->done = 1; p->st.attempts++; }
    }
}

/* The decisions of mynah_asr_recover_turn (src/recover.c) over the attempts'
 * finished slots. */
static void job_decide(rpool *p, rp_job *jb, rpool_result *r) {
    const char *texts[MYNAH_ASR_RECOVER_MAX_ATTEMPTS];
    char langs[MYNAH_ASR_RECOVER_MAX_ATTEMPTS][16];
    for (int a = 0; a < jb->n; a++) {
        const rp_attempt *at = &jb->att[a];
        const int *tok = NULL, *fr = NULL;
        const int nt = asr_engine_slot_tokens(p->e, at->slot, &tok, &fr);
        char *text = NULL;
        if (nt >= 0) {
            const int k = mynah_asr_replay_keep_from(p->tk, tok, fr, nt, at->a.prefix_frames,
                                                     p->c.cfg.rule, jb->prev_text, jb->lag);
            text = mynah_asr_detokenize(p->tk, tok + k, nt - k, NULL);
        }
        if (!text) text = strdup("");   /* no frames from the engine: no vote */
        snprintf(langs[a], sizeof(langs[a]), "%s", asr_engine_slot_lang(p->e, at->slot));
        if (text && mynah_asr_recovery_gate(text, jb->lang, langs[a], jb->prev_text) != MYNAH_ASR_GATE_OK)
            text[0] = '\0';   /* a rejected attempt keeps no vote */
        p->texts[a] = text;
        texts[a] = text;
        r->attempts++;
    }
    const int w = mynah_asr_recover_vote(texts, jb->n, p->c.cfg.min_votes);
    if (w < 0) { r->status = RP_REJECTED; p->st.rejected++; return; }
    r->status = RP_ACCEPTED;
    r->winner = w;
    r->kind = jb->att[w].a.kind;
    r->text = texts[w];
    for (int a = 0; a < jb->n; a++)
        r->votes += texts[a] && mynah_asr_text_has_content(texts[a]) && mynah_asr_text_same_words(texts[w], texts[a]);
    snprintf(r->lang, sizeof(r->lang), "%s", langs[w]);
    if (r->kind == MYNAH_ASR_ATTEMPT_REPLAY) p->st.accepted_replay++; else p->st.accepted_phase++;
}

int rpool_next_result(rpool *p, double now, rpool_result *r) {
    for (int i = 0; i < MYNAH_ASR_RECOVER_MAX_ATTEMPTS; i++) { free(p->texts[i]); p->texts[i] = NULL; }
    int order[RPOOL_MAX_SLOTS];
    if (p->c.n_slots > RPOOL_MAX_SLOTS) return 0;
    const int nj = job_order(p, order);
    for (int o = 0; o < nj; o++) {
        const int j = order[o];
        rp_job *jb = &p->jobs[j];
        int all = 1;
        for (int a = 0; a < jb->n; a++) all &= jb->att[a].done;
        const int late = p->c.deadline_s > 0.0 && now - jb->t0 > p->c.deadline_s;
        if (!all && !late) continue;
        memset(r, 0, sizeof(*r));
        r->job = j; r->owner = jb->owner; r->gen = jb->gen;
        r->winner = -1; r->text = "";
        r->age_s = now - jb->t0;
        if (all) job_decide(p, jb, r);
        else { r->status = RP_DEADLINE; p->st.deadline++; }
        job_free(p, j);
        return 1;
    }
    return 0;
}

int rpool_pending(const rpool *p) {
    int n = 0;
    for (int j = 0; j < p->c.n_slots; j++) n += p->jobs[j].used;
    return n;
}

void rpool_get_stats(const rpool *p, rpool_stats *s) { *s = p->st; }

const char *rpool_kind_name(mynah_asr_attempt_kind k) {
    return k == MYNAH_ASR_ATTEMPT_REPLAY ? "replay" : "phase";
}
