# S2 — WebSocket protocol v2

Status: IMPLEMENTED except the v1 compatibility fields (see Unknowns)

Task: S2-5
Question: a wire protocol that exposes what the library already knows
(final/eou/timing), lets a client control the session, and lets both sides
measure lag honestly.

Known facts (v1)
- `GET /v1/audio/stream?lang=&lookahead=`; binary s16le 16 kHz frames in;
  `{"text","language","audio_seconds"}` out; `is_eou`/`is_final` dropped;
  frames > 256 KiB drop the connection; no control channel; no pings; no
  sequence numbers; close = finish.

v2 (see serving-v2-design.md §5)
- Query adds `model=`, `format=s16le|f32le`, `rate=` (resampled if not 16000,
  reported in the first frame). Unknown query keys → 400 naming the accepted set.
- Client text frames: `{"type":"finalize"}` (flush the tail, emit `done`, keep
  the socket for a new utterance after `reset`), `{"type":"reset"}`.
- Server frames all carry `seq`, `audio_s`, `lag_ms`:
  `delta{text, final:true, t0, t1, lang}`, `eou{t}`, `done{lang, audio_s}`,
  `error{code, message}`. Nemotron deltas are always `final:true`; the field
  exists so an unstable-tail decoder can later send `final:false` without a
  protocol change.
- Refusals before the 101 as HTTP statuses (S2-3).
- `tools/eval/ws_client.py` (stdlib) updated and used by `tests/test_server.sh`.

Gate: `docs/server.md` protocol section rewritten; a protocol test drives
finalize/reset twice on one socket and checks the second utterance's text
equals a fresh connection's; unknown message types get an `error` frame, not
a disconnect.

## Evidence

Host: Apple M1 (arm64, 8 cpus), macOS 25.5, Accelerate, `make` default flags,
`nemotron-3.5-asr-streaming-0.6b --quant int8`. Development-machine numbers.

### What was built

- **Query parsing is a parser, not three `strstr`s.** v1 read `lang=` and
  `lookahead=` with `strstr`, so `?xlang=de` set the language and `?lookahed=3`
  was silently ignored — a client with a typo got a stream that was not the one
  it asked for. `ws_parse_query()` splits on `&`/`=` and validates every key and
  value against the model's own config *before* the upgrade:

  | key | accepted | refusal |
  |---|---|---|
  | `model` | what `/v1/models` lists | `404 model_not_found` |
  | `lang` | the model's prompt dictionary | `400 language_not_served` |
  | `lookahead` | the model's presets (0, 3, 6, 13 for Nemotron) | `400 lookahead_not_available` |
  | `format` | `s16le`, `f32le` | `400 unsupported_format` |
  | `rate` | `16000` | `400 unsupported_rate` |
  | anything else | — | `400 unknown_query_parameter`, listing the set |

  The lookups (`mynah_asr_lang_id`, `mynah_asr_lookaheads`) are reads of the
  model's config, not inference, so they stay on the HTTP thread and cost the
  scheduler nothing — the same justification `handle_transcribe` already used
  for `mynah_asr_map_lang`.
- **`rate=` refuses instead of resampling, deliberately.** `mynah_asr_resample`
  is a whole-buffer, stateless windowed sinc: no filter history is carried
  between calls, so resampling per WebSocket frame would inject a discontinuity
  at every frame boundary and change the transcript in a way no identity gate
  would catch. The REST path resamples because it holds the whole file. A
  streaming resampler is separate work with its own parity gate; until then the
  honest answer is a 400 naming what is served. (This is the one place where
  the plan above says "resampled if not 16000" and the code says no; the reason
  is recorded here rather than the plan quietly followed.)
- **`format=f32le`** is decoded byte-wise like s16le: the payload sits wherever
  the WebSocket header left it, and reading it through a `float *` is an
  unaligned access — the same UB ubsan already caught in the v1 `int16_t *` cast.
- **Unknown control `type` → `error` code `unknown_control`** (v1 called it
  `unknown_message`), session survives.
- **`reset` with an unserved `lang` → `error language_not_served`**, session
  survives. Previously the bad tag travelled to the scheduler, failed
  `mynah_asr_stream_reset` and cancelled the stream with `reset_failed`: a typo
  cost a session. The check is now a dictionary lookup on the ingest thread.
- **Server pings** every `--ping-ms` (default 20000) through `stream_out`, from
  the ingest thread's existing 200 ms tick. No pong is required.
- **`finalize` then more audio** already worked (the scheduler sets
  `needs_reset` and `seq` keeps counting); what was missing was a test.

### What ran, and what it measured

`tests/test_server_protocol.sh` (new, model-gated, wired into `make test` and
`make test-server-protocol`) with `tests/ws_probe.py` (new, stdlib, one
self-checking probe per behaviour). Run against `nemotron --quant int8`:

```
server-protocol three-utterances-one-socket OK   seq 0..37 over 3 utterances
server-protocol unknown-control             OK
server-protocol reset-bad-lang              OK
server-protocol unknown-query-key           OK   400, accepted set in the body
server-protocol unsupported-rate            OK   400, "accepted: 16000"
server-protocol audio-limit                 OK   --max-audio-seconds 1, finalised
server-protocol idle-timeout                OK
server-protocol server-pings                OK   2 pings in 3 s at --ping-ms 1000
server-protocol cap-1-second-stream-503     OK   503 + Retry-After, readable body
server-protocol sigterm-shutting-down-frame OK
server-protocol sigterm-exit-0              OK
server-protocol sigterm-no-survivors        OK
```

The identity assertion is stronger than the note's gate asked for: not "the
second utterance equals a fresh connection's" but **each of three utterances on
one socket is byte-identical to `./mynah-asr transcribe --quant int8 --lang
auto` of its clip**, with `seq` strictly increasing 0..37 across all three. The
reference is built in the test from the same binary, so a stale file cannot
pass for a measurement.

`tests/test_server.sh` (which reads the v1 fields through
`tools/eval/ws_client.py`), `tests/test_server_stream.sh` and
`tests/test_server_concurrency.sh` are all still 0, and all four are 0 under a
`-fsanitize=undefined` build with 0 runtime-error lines (see
`.work/server-admission.md` for the flags, and for why the sanitizer must not be
stacked on `-ffast-math`).

## Conclusion

The control channel, the validated query and the refusals are in and gated. The
protocol is v2 on the wire except that `delta` and `done` still carry their v1
aliases, so S2-5 is not closed.

## Unknowns

- **The v1 compatibility fields are still on the wire**: `delta` carries
  `language` and `audio_seconds` beside `lang` and `audio_s`, and `done` carries
  `done:true`, `language` and `audio_seconds`. `tools/bench/stream_load.py` and
  `tools/eval/ws_client.py` already read the v2 names with the v1 ones as a
  fallback, so removing them is a small change — it was left out of this pass
  because it is a compatibility break that nothing in the gate required, and
  breaking a wire format as a side effect of adding a test is how a release
  note ends up wrong.
- `delta` carries `audio_s` (the end of the chunk) but not the `t0`/`t1` pair
  the design note lists. Nothing reads them yet.
- `--max-audio-seconds` is counted over the connection, not per utterance.
- No Linux run of the protocol test.

## Next action

Drop the v1 aliases and add `t0`/`t1` to `delta` (one commit, one release note),
then S2-6.
