/* Mynah — a lightweight native C runtime for streaming and offline ASR.
 *
 * API pubblica di libmynah. Fase attuale: trascrizione offline-chunked (M1.2).
 * Streaming API (mynah_stream_*) in arrivo con M1.3.
 */
#ifndef MYNAH_H
#define MYNAH_H

#include <stddef.h>
#include <stdbool.h>

#define MYNAH_VERSION_MAJOR 0
#define MYNAH_VERSION_MINOR 6
#define MYNAH_VERSION_PATCH 0
#define MYNAH_VERSION "0.6.0-dev"

#ifdef __cplusplus
extern "C" {
#endif

/* Risultato incrementale di trascrizione (streaming e offline).
 * text è UTF-8, valido solo per la durata della callback. */
typedef struct {
    const char *text;      /* testo del segmento/parziale                   */
    double      t0, t1;    /* finestra temporale in secondi (se disponibile) */
    bool        is_final;  /* false = partial (può cambiare), true = commit  */
    const char *lang;      /* tag lingua rilevata, NULL se non disponibile   */
} mynah_result;

typedef void (*mynah_result_cb)(const mynah_result *res, void *userdata);

const char *mynah_version(void);

/* ----------------------------------------------------------------- modello */
typedef struct mynah_model mynah_model;

/* Carica un modello convertito (directory con mynah.json + model.safetensors
 * + tokens.json + mel_filters.safetensors). NULL su errore. */
mynah_model *mynah_load(const char *model_dir);

/* Come mynah_load con quantizzazione: MYNAH_QUANT_INT8 costruisce al load l'INT8
 * per-riga sui grandi linear (~2.4x meno RAM, qualita' quasi identica). */
enum { MYNAH_QUANT_F32 = 0, MYNAH_QUANT_INT8 = 1, MYNAH_QUANT_INT4 = 2 };
mynah_model *mynah_load_quant(const char *model_dir, int quant);
void mynah_free(mynah_model *m);

/* Risolve un tag lingua ("it-IT", "auto", ...) nel prompt id. -1 se ignoto. */
int mynah_lang_id(const mynah_model *m, const char *lang);

/* Seleziona il decoder per le trascrizioni successive: "default" (RNNT/TDT del
 * modello) o "ctc" (head ausiliaria dei modelli hybrid, più veloce, solo offline).
 * -1 se il modello non supporta il decoder richiesto. */
int mynah_set_decoder(mynah_model *m, const char *name);

/* Solo modelli AED (Canary): lingua di USCITA delle trascrizioni successive
 * ("en", "de", ...). Diversa dalla sorgente = speech translation. NULL o "" =
 * uguale alla sorgente (ASR). -1 se non supportata dal modello.
 * Alternativa PER-CHIAMATA (thread-safe, per server/batch): lang "src>tgt"
 * in mynah_transcribe* (es. "en>de"); vince su mynah_set_target_lang. */
int mynah_set_target_lang(mynah_model *m, const char *lang);

/* 1 se il modello supporta la speech translation (engine AED). */
int mynah_can_translate(const mynah_model *m);

/* Limite di durata per segmento nella trascrizione offline: gli audio più lunghi
 * vengono divisi sul minimo di energia (silenzio) vicino al confine e trascritti
 * a segmenti indipendenti (testo e timestamp concatenati). Default MODEL-AWARE:
 * 30 s per i modelli full-attention/AED (Parakeet, Canary — addestrati su
 * utterance corte: segmenti lunghi degradano la qualità), 300 s per i modelli
 * ad attention finestrata (Nemotron). sec >= 5; 0 = ripristina il default. */
void mynah_set_segment_limit(mynah_model *m, double sec);

/* Lookahead (right context) validi per il modello, es. {3,0,6,13}. */
int mynah_lookaheads(const mynah_model *m, int out[8]);

/* Trascrizione offline: samples float32 [-1,1] 16 kHz mono.
 * lang: tag ("auto" per detection). lookahead: -1 = default del modello.
 * Ritorna testo UTF-8 (malloc, caller free); se lang_out != NULL (>= 16 byte)
 * vi scrive la lingua rilevata. NULL su errore. */
char *mynah_transcribe(mynah_model *m, const float *samples, size_t n_samples,
                       const char *lang, int lookahead, char *lang_out);

/* Parola con finestra temporale (dai frame encoder di emissione greedy:
 * risoluzione = 1 frame encoder, tipicamente 80 ms). */
typedef struct {
    char  *word;           /* UTF-8, senza marcatori (malloc) */
    double t0, t1;         /* secondi dall'inizio dell'audio   */
} mynah_word;

/* Come mynah_transcribe, in piu' scrive in *words un array malloc di *n_words
 * parole con timestamp (liberare con mynah_words_free). words == NULL = solo testo. */
char *mynah_transcribe_ts(mynah_model *m, const float *samples, size_t n_samples,
                          const char *lang, int lookahead, char *lang_out,
                          mynah_word **words, int *n_words);

void mynah_words_free(mynah_word *words, int n_words);

/* Trascrizione BATCH weight-stationary: N richieste processate insieme — i pesi
 * (2.5 GB) vengono letti una volta per layer invece di N. Lunghezze variabili,
 * nessun padding. texts[i] riceve il testo (malloc, caller free; NULL su errore
 * del singolo item); langs_out[i] (>= 16 byte l'uno) opzionale. 0 = ok. */
/* Come mynah_transcribe_batch, in piu' words[b]/n_words[b] per item (array di
 * `batch` slot forniti dal chiamante; ogni words[b] va liberato con
 * mynah_words_free). words == NULL = solo testo. */
int mynah_transcribe_batch_ts(mynah_model *m, const float *const *samples,
                              const size_t *n_samples, int batch, const char *const *langs,
                              int lookahead, char **texts, char (*langs_out)[16],
                              mynah_word **words, int *n_words);
int mynah_transcribe_batch(mynah_model *m, const float *const *samples,
                           const size_t *n_samples, int batch, const char *const *langs,
                           int lookahead, char **texts, char (*langs_out)[16]);

/* --------------------------------------------------------------- streaming
 * Cache-aware, latenza = (lookahead+1) * 80 ms. Il testo emesso via callback è
 * SEMPRE definitivo (greedy monotono, mai ritrattato): is_final = true. */
typedef struct mynah_stream mynah_stream;

mynah_stream *mynah_stream_open(mynah_model *m, const char *lang, int lookahead);

/* Alimenta campioni float32 16 kHz mono; la callback riceve i delta di testo. */
int mynah_stream_feed(mynah_stream *s, const float *samples, size_t n,
                      mynah_result_cb cb, void *userdata);

/* Fine stream: processa la coda (ultimo chunk paddato) ed emette il resto. */
int mynah_stream_finish(mynah_stream *s, mynah_result_cb cb, void *userdata);

/* Lingua rilevata finora ("" se non ancora emessa). */
const char *mynah_stream_lang(const mynah_stream *s);

void mynah_stream_close(mynah_stream *s);

#ifdef __cplusplus
}
#endif

#endif /* MYNAH_H */
