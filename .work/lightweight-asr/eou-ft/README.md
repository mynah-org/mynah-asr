# parakeet_realtime_eou_120m-v1 -> Italian specialist (streaming RNNT + `<EOU>`)

Recipe facts, file:line references and what is real vs synthetic in the EOU labels:
`.work/eou-it-ft.md`. Reuses the Canary kit's data on the box (`/root/ft/manifests`,
`/root/ft/text/it_pool.txt`, built by `../ft/data_it.sh`) and its `ftlib.py`
(normaliser, scoring, level re-quantisation, gain augmentation).

| file | what |
|---|---|
| `tokenizer_eou_it.py` | Italian SPE (type/spec copied from the stock tokenizer, vocab 1024) + `<EOU>`/`<EOB>` appended as NeMo's `add_special_tokens_to_sentencepiece.py` does; verified 1024/1025, blank 1026 |
| `train_eou_it.py` | stock encoder + fresh decoder/joint, `EncDecRNNTBPEEOUModel` lhotse EOU dataset (zero padding, white noise), gain -30..+6 dB, bf16; metrics.json (economics, IT WER/CER/empty/EOU-rate at native and -3/-20/-40 dBFS, EN info, NeMo cache-aware streaming check); final.nemo saved as plain `EncDecRNNTBPEModel` |
| `export_to_mynah.sh` | `tools/convert_nemo.py` -> pack, token/preset checks, `mynah-asr stream --deltas` on 3 clips, prints the `lang_gate.py` / `eou_metrics.py` commands for an Italian bank |
| `../jobs/eou1.sh` | download (resumable) + tokenizer outside the GPU lock; train/eval/export under `flock /root/gpu.lock`; summary |

```bash
scp -r .work/lightweight-asr/eou-ft root@BOX:/root/eou-kit
scp .work/lightweight-asr/ft/ftlib.py root@BOX:/root/eou-kit/
scp .work/lightweight-asr/jobs/eou1.sh root@BOX:/root/eou1.sh
ssh root@BOX "RATE_USD_H=<price> tmux new -d -s eou1 'bash /root/eou1.sh 2>&1 | tee -a /root/ft/logs/eou1.log'"
```

Laptop checks: `python3 tokenizer_eou_it.py --dry-run --out /tmp/tok`;
`FT_ROOT=<tree with manifests/train_5h.json> python3 train_eou_it.py --dry-run`.
