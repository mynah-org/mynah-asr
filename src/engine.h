/* Interfaccia engine: un tipo di decoder = un engine dietro la vtable.
 *
 * L'encoder resta un percorso UNICO condiviso (regola repo #2: niente percorsi
 * divergenti); l'engine decodifica l'output encoder di UN segmento in token.
 * Lo streaming cache-aware sta fuori dalla vtable: è incrementale per natura
 * (stato per-chunk) e oggi esiste solo per Nemotron.
 *
 * Per aggiungere un engine (es. Whisper-style, Moonshine): una mynah_asr_engine
 * statica + il suo decode, selezionata al load dal decoder.type del mynah.json.
 */
#ifndef MYNAH_ASR_ENGINE_H
#define MYNAH_ASR_ENGINE_H

struct mynah_asr_model;

typedef struct {
    const char *name;
    /* 1: l'engine consuma l'encoder out PRE-projector (mynah_asr_encoder_forward_raw,
     * es. head CTC); 0: output post prompt/projector (mynah_asr_encoder_forward). */
    int raw_encoder;
    /* Decodifica enc [T, d] -> *tokens (malloc del chiamato, caller free).
     * Ritorna n token >= 0, -1 su errore.
     * lang: tag della richiesta (usato dal prompt AED, "src>tgt" = traduzione).
     * want_ts: il chiamante vuole i timestamp — l'engine scrive in *frames un
     * array malloc parallelo ai token (frame encoder di emissione) oppure lo
     * lascia NULL se produce i tempi in altro modo (AED: token <|N|> dentro
     * la sequenza, estratti a valle da aed_words_from_tokens). */
    int (*decode)(struct mynah_asr_model *m, const float *enc, int T,
                  const char *lang, int want_ts, int **tokens, int **frames);
} mynah_asr_engine;

#endif
