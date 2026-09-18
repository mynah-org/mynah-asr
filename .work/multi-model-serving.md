# S2 — several models in one fleet

Status: OPEN

Task: S2-6
Question: serve Nemotron (streaming, 40 languages), Parakeet TDT (offline, 25
languages) and Canary (offline, translation) from one `mynah-asr-server`
invocation, with routing by model and language per request.

Known facts
- One process holds one model: a batch is one model by the shape of the
  address space (sibling decision B; the in-process registry was costed and
  rejected there because switching weights per request costs a whole pass).
- Unlike the TTS siblings, **language is per request here**: Nemotron's prompt
  is a per-row post-encoder one-hot, Canary's is a per-item prompt. A batch
  may mix languages. Routing is by model only; `lang` travels with the request.
- `--lid-model` already exists: a detector model in front of models that
  cannot detect. In v2 the detector is its own worker group and the router
  asks it first only for `lang=auto` on a group that cannot detect (Canary).
- Router classification is a bounded non-consuming `MSG_PEEK` ≤ 8 KiB for the
  request line (query `model=`) and, for multipart REST, the `model` field if
  it is inside the peeked prefix; otherwise the default group. Ambiguity is
  refused, never resolved.

CLI shape
```
mynah-asr-server \
  --model nemotron=models/nemotron-3.5-asr-streaming-0.6b:workers=6:cpus=24:cap=8:quant=int8 \
  --model parakeet=models/parakeet-tdt-0.6b-v3:workers=2:cpus=8:cap=4:quant=int8 \
  --default nemotron -p 8090
```
`/v1/models` lists the groups from the actual table (fixing the hard-coded
name). A group whose model cannot stream answers WS with 400
`model_not_streaming`.

Gate: three groups up on the Axion; REST to each; WS to Nemotron; the byte-
identity gate per group; `/v1/models` matches the CLI; a request naming an
unknown model gets 404 with the OpenAI error shape.

Evidence / Conclusion / Next action: after S2-1..S2-5.
