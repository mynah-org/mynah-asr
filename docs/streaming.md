# Mynah — Streaming: semantics and API

## Mental model

Nemotron is a **cache-aware** encoder: every audio sample is processed exactly
once. The stream is a sequence of **fixed-size mel chunks**; each chunk produces
`q = lookahead+1` encoder frames (1 frame = 80 ms of audio) and the emitted text is
**always final** — greedy RNNT is monotonic, it never retracts.

| lookahead | theoretical latency | mel chunk (first / subsequent) |
|---|---|---|
| 0 | 80 ms | 1 / 8 |
| 1 | 160 ms | 9 / 16 |
| 3 (default) | 320 ms | 25 / 32 |
| 6 | 560 ms | 49 / 56 |
| 13 | 1.12 s | 105 / 112 |

Rule: first chunk = `1 + 8·lookahead` mel frames, subsequent = `8·(lookahead+1)`.
The lookahead is chosen **at runtime** (`--lookahead` / API parameter), without
reloading the model: quality and latency scale together (see per-preset WER in the
model card — at 80 ms the first words may degrade).

**Equivalence guarantee**: streaming produces *exactly* the same text as the
offline path on the same audio (`make test`, `tests/test_streaming.c` verifies
byte-for-byte equality). The stream tail is handled with a short chunk +
causal right-pad, identical to the offline math.

## Internal state (for the curious / debugging)

- **Incremental mel**: O(n_fft) sliding window of pre-emphasized signal;
  each mel frame is bit-identical to offline (possible because the model does not
  normalize features).
- **Subsampling**: cache of 1 input frame for each of the 3 conv stages
  (+1 init zero on the first chunk = left pad 2 offline).
- **Attention**: per-layer K/V cache `[56, d_model]`. The chunk coincides with the
  `chunked_limited` grid and 56 is divisible by every `q` → the cache contains
  exactly the allowed context and the attention is dense, no mask.
- **Conv module**: per-layer cache `[8, d_model]` (causal kernel 9).
- **Decoder**: lifted, chunk-invariant LSTM state — incremental decoding ≡
  decoding the entire audio.

## C API

```c
mynah_asr_model *m = mynah_asr_load("models/nemotron-3.5-asr-streaming-0.6b");
mynah_asr_stream *s = mynah_asr_stream_open(m, "auto" /* or "it-IT", ... */, 3 /* lookahead */);

void on_text(const mynah_asr_result *r, void *ud) {
    fputs(r->text, stdout);          /* text delta, already final */
    /* r->lang = detected language (with "auto"), r->t1 = seconds of audio consumed */
}

while (have_audio) {
    /* float32 [-1,1], 16 kHz mono, any feed size */
    mynah_asr_stream_feed(s, samples, n, on_text, NULL);
}
mynah_asr_stream_finish(s, on_text, NULL);   /* process the tail */
mynah_asr_stream_close(s);
mynah_asr_free(m);
```

Notes:
- `feed` accepts any number of samples; internal chunking is automatic.
- With `lang="auto"` the language tag arrives when the model emits it
  (`mynah_asr_stream_lang()` to read it at any time).
- Memory cost per stream: ~12 MB of cache (24 layers × K/V 56×1024 + conv).
- Realtime: ~26 ms of compute per 80 ms chunk on Apple Silicon (~3× headroom).

## CLI

```sh
# microphone (sox example) → partials on stdout as they arrive
rec -q -t raw -r 16000 -e signed -b 16 -c 1 - | mynah-asr stream -m <model_dir> --lang auto

# file, via ffmpeg
ffmpeg -v quiet -i audio.mp3 -f s16le -ar 16000 -ac 1 - | mynah-asr stream -m <model_dir>
```

## Endpointing (end of utterance)

With a VAD attached to the model, the stream also tells you **where utterances
end** — useful for committing a turn, driving a UI, or deciding when to reply:

```c
mynah_asr_enable_vad(m, "models/silero-vad");   /* before opening the stream */
mynah_asr_stream *s = mynah_asr_stream_open(m, "auto", -1);
/* in the callback: */
if (res->is_eou) commit_turn(res->t1);          /* text is "", t1 = speech stopped */
```

Three properties worth knowing, all of them measured rather than assumed:

- **It does not change the transcription.** Endpointing only observes; the text is
  byte-identical with and without a VAD. `tests/test_e2e.sh` fails if it is not.
- **Latency is ~160 ms of audio plus one feed chunk.** The 160 ms is
  `min_silence_ms` (100) plus frame quantization: the VAD cannot know speech ended
  until enough silence has passed. Feeding smaller buffers lowers the rest —
  measured on nemotron: 160 ms with 32 ms feeds, 176-212 ms with 100 ms, 312-376 ms
  with 250 ms.
- **The endpoint can arrive before the last words of that utterance.** The VAD runs
  ahead of the encoder, which works in chunks. If you need the full text of a turn,
  wait for the deltas that follow the endpoint (or `finish` at the end of the
  stream), rather than treating the endpoint as "text complete".

Each stream keeps its own VAD instance, so concurrent streams do not interfere
(the VAD carries LSTM state; sharing it would mix two conversations).

```sh
# a WAV through the streaming path, 32 ms at a time, with endpoints on stderr
mynah-asr stream -m <model_dir> -i audio.wav --chunk-ms 32 --vad models/silero-vad
```
