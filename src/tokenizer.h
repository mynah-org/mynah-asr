/* SentencePiece-BPE detokenizer: ids -> pieces -> UTF-8 text.
 * Loads tokens.json (the array of pieces produced by the converter).
 * The special <xx-XX> tokens are language tags: stripped and reported as lang. */
#ifndef MYNAH_ASR_TOKENIZER_H
#define MYNAH_ASR_TOKENIZER_H

typedef struct {
    char **pieces;
    int n_pieces;
} mynah_asr_tokenizer;

int mynah_asr_tokenizer_load(mynah_asr_tokenizer *tk, const char *tokens_json_path);
void mynah_asr_tokenizer_free(mynah_asr_tokenizer *tk);

/* id of the FIRST occurrence of the piece (aggregate tokenizers repeat <unk> in
 * each sub-vocab: the specials live in the first one). -1 when absent. */
int mynah_asr_tok_find(const mynah_asr_tokenizer *tk, const char *piece);

/* Decode tokens -> text (malloc'd, freed by the caller). When a language tag is
 * present it writes up to 15 chars into lang_out (when non-NULL). ▁ -> space,
 * leading space dropped. */
char *mynah_asr_detokenize(const mynah_asr_tokenizer *tk, const int *tokens, int n,
                       char *lang_out);

#include "mynah_asr.h"  /* mynah_asr_word */

/* Group tokens into words (a new word at every piece starting with ▁) with
 * timestamps from the emission frames: t0 = frame of the first piece * frame_sec,
 * t1 = (frame of the last piece + 1) * frame_sec. Special <...> tokens are
 * skipped. Writes a malloc'd array into *out (caller: mynah_asr_words_free).
 * 0 = ok. */
int mynah_asr_detokenize_words(const mynah_asr_tokenizer *tk, const int *tokens,
                           const int *frames, int n, double frame_sec,
                           mynah_asr_word **out, int *n_out);

#endif
