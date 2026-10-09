#!/usr/bin/env python3
"""Markdown summary of everything the kit measured under $FT_ROOT (stdout)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ftlib  # noqa: E402

FT = ftlib.FT


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def main():
    h = load(FT / "hours.json")
    if h:
        print("### Data (hours from the written wav files)\n")
        for k, v in h.items():
            if isinstance(v, dict) and "hours" in v:
                print(f"- {k}: {v['n_utts']} utts, {v['hours']} h (missing audio {v['missing_audio']})")
        print()
    s = load(FT / "tokenizers" / "surgery_report.json")
    if s:
        print(f"### Tokenizer surgery\n\n- vocab {s['v_old']} -> {s['v_new']} (it offset {s['it_offset']}); "
              f"old rows bit-identical {s['old_rows_bit_identical']}; EN identical {s['en_identical']}; "
              f"it rows init {s['it_row_init']}\n")
    z = load(FT / "probe" / "zeroshot.json")
    if z:
        print("### C0 zero-shot (untouched model, Italian audio)\n\n| prompt | set | WER | CER | S/D/I | empty |\n|---|---|---|---|---|---|")
        for arm, sets in z["arms"].items():
            for name, sc in sets.items():
                w = sc["words"]
                print(f"| {arm}/{arm} | {name} | {sc['wer']} | {sc['cer']} | {w['S']}/{w['D']}/{w['I']} | {sc['empty_hyp']} |")
        print(f"\nEN sanity (FLEURS en_us, {z['en']['fleurs_en']['n_utts']} clips): WER {z['en']['fleurs_en']['wer']}; "
              f"re-levelled: " + ", ".join(f"peak {lv} dBFS {v['fleurs_en']['wer']}" for lv, v in (z.get("en_level_sweep") or {}).items())
              + f"; preprocessor {z.get('preprocessor')}\n")
    runs = sorted((FT / "runs").glob("*/metrics.json"))
    if runs:
        print("### Fine-tune runs\n\n| run | subset h | steps | epochs | FLEURS-it WER | MLS-it WER | EN WER before->after | "
              "audio-h/GPU-h | samples/s | peak alloc/res GB | epoch wall s | GPU-h total | cost USD |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for p in runs:
            m = load(p)
            f = m.get("final_it") or {}
            eb, ea = m["en_before"]["fleurs_en"]["wer"], m["en_after"]["fleurs_en"]["wer"]
            print(f"| {m['tag']} | {m['subset_hours_exact']} | {m['steps_done']} | {m['equiv_epochs']} | "
                  f"{f.get('fleurs_it', {}).get('wer')} | {f.get('mls_it', {}).get('wer')} | {eb} -> {ea} | "
                  f"{m['audio_h_per_gpu_h']} | {m['samples_per_s']} | {m['peak_alloc_gb']}/{m['peak_reserved_gb']} | "
                  f"{m['epoch_wall_s']} | {m['gpu_hours_total']} | {m['cost_usd_total']} (@{m['rate_usd_h']}/h) |")
        print()
        print("### Level sweep (clips re-levelled to a fixed peak; IT = pooled val subsets)\n\n"
              "| run | gain aug | peak dBFS | IT WER before -> after | EN WER before -> after |\n|---|---|---|---|---|")
        for p in runs:
            m = load(p)
            ls = m.get("level_sweep") or {}
            b, f = ls.get("before") or {}, ls.get("after") or {}
            for lv in (b.get("it_val") or {}):
                print(f"| {m['tag']} | {'on' if m.get('gain_aug') else 'off'} | {lv} | "
                      f"{b['it_val'][lv]['_pooled']['wer']} -> {f['it_val'][lv]['_pooled']['wer']} | "
                      f"{b['en'][lv]['fleurs_en']['wer']} -> {f['en'][lv]['fleurs_en']['wer']} |")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
