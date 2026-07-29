/* Parallel-for minimale su pthread per i loop CPU indipendenti (frame mel,
 * canali depthwise, segmenti del batch). I task devono scrivere su regioni
 * disgiunte: il risultato è BIT-IDENTICO al loop seriale per costruzione
 * (stesso codice per task, solo su thread diversi).
 * Numero thread: env MYNAH_ASR_THREADS, default = core online. */
#ifndef MYNAH_ASR_THREADS_H
#define MYNAH_ASR_THREADS_H

int mynah_asr_num_threads(void);

/* Esegue fn(ctx, i) per i in [0, n): i task vengono distribuiti su
 * min(n, mynah_asr_num_threads()) thread (il chiamante partecipa).
 * Con n <= 1 o un solo thread gira in-place senza spawn. */
void mynah_asr_parallel_for(int n, void (*fn)(void *ctx, int i), void *ctx);

#endif
