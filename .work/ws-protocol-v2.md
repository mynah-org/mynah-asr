# S2 — WebSocket protocol v2

Status: OPEN

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

Evidence / Conclusion / Next action: with S2-2.
