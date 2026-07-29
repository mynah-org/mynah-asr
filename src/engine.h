/* Engine interface: one decoder type = one engine behind the vtable.
 *
 * The encoder stays a SINGLE shared path (repo rule #2: no divergent paths);
 * the engine decodes the encoder output of ONE segment into tokens.
 * Cache-aware streaming lives outside the vtable: it is incremental by nature
 * (per-chunk state) and today exists only for Nemotron.
 *
 * To add an engine (e.g. Whisper-style, Moonshine): one static mynah_asr_engine
 * plus its decode, selected at load time from decoder.type in mynah.json.
 */
#ifndef MYNAH_ASR_ENGINE_H
#define MYNAH_ASR_ENGINE_H

struct mynah_asr_model;

typedef struct {
    const char *name;
    /* 1: the engine consumes the PRE-projector encoder output
     * (mynah_asr_encoder_forward_raw, e.g. the CTC head); 0: the post
     * prompt/projector output (mynah_asr_encoder_forward). */
    int raw_encoder;
    /* Decode enc [T, d] -> *tokens (malloc'd by the callee, freed by the caller).
     * Returns the token count >= 0, or -1 on error.
     * lang: the request tag (used by the AED prompt, "src>tgt" = translation).
     * want_ts: the caller wants timestamps — the engine either writes into
     * *frames a malloc'd array parallel to the tokens (the encoder frame each
     * token was emitted at) or leaves it NULL when it produces times some other
     * way (AED: <|N|> tokens inside the sequence, extracted downstream by
     * aed_words_from_tokens). */
    int (*decode)(struct mynah_asr_model *m, const float *enc, int T,
                  const char *lang, int want_ts, int **tokens, int **frames);
} mynah_asr_engine;

#endif
