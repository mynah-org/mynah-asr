#!/usr/bin/env python3
"""Multilingual suite: transcribes the per-language samples (tests/audio/langs/) with
`mynah-asr transcribe --lang auto` and checks (a) language detection, (b) CER vs the
reference text (normalized: lowercase, no punctuation).

Usage: uv run python -m eval.test_langs [--cer-max 0.3] [--mynah-asr ../mynah-asr] [--model DIR]
                                      [--quant int8|int4]
--quant: quantization regression — the same suite over the pre-quantized
checkpoint (requires model.int8/int4.safetensors from `mynah-asr quantize`).
Exit: 0 every language ok, 1 failures, 77 skip (samples or model missing).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# The model card's "adaptation-ready" tier: weak quality BY DESIGN (it needs
# fine-tuning). Reported as WEAK; they do not count as suite failures.
ADAPTATION_TIER = {"el-GR", "lt-LT", "lv-LV", "mt-MT", "sl-SI", "he-IL", "th-TH", "nn-NO"}

# Regional variants (different prompt ids, same language): tested by reusing the
# samples of the base language with the variant's prompt. 40 locales = 36 languages.
VARIANTS = {"en-GB": "en-US", "es-US": "es-ES", "fr-CA": "fr-FR", "pt-PT": "pt-BR"}


def normalize(s: str) -> str:
    s = re.sub(r"<[^<>]{1,12}>", " ", s)   # language tag spelled out by the model
    s = unicodedata.normalize("NFKC", s).lower()
    s = "".join(c for c in s if not unicodedata.category(c).startswith("P"))
    return re.sub(r"\s+", " ", s).strip()


def cer(ref: str, hyp: str) -> float:
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cer-max", type=float, default=0.3)
    ap.add_argument("--mynah-asr", dest="mynah_asr", default=str(ROOT / "mynah-asr"))
    ap.add_argument("--model", default=str(ROOT / "models/nemotron-3.5-asr-streaming-0.6b"))
    ap.add_argument("--quant", choices=["int8", "int4"], default=None)
    ap.add_argument("--only", default=None,
                    help="filter the locales (csv, e.g. it-IT,de-DE) — for models "
                         "supporting a subset of the languages; implies strict tier")
    args = ap.parse_args()

    manifest_path = ROOT / "tests/audio/langs/manifest.json"
    if not manifest_path.exists() or not Path(args.model, "mynah.json").exists():
        print("SKIP: samples (tools/fetch_lang_samples.py) or model missing")
        sys.exit(77)
    if args.quant and not Path(args.model, f"model.{args.quant}.safetensors").exists():
        print(f"SKIP: {args.quant} checkpoint missing (mynah-asr quantize --quant {args.quant})")
        sys.exit(77)
    manifest = json.loads(manifest_path.read_text())
    model_cfg = json.loads(Path(args.model, "mynah.json").read_text())
    has_prompt = "prompt" in model_cfg   # without a prompt (Parakeet): --lang ignored, implicit LID
    only = set(args.only.split(",")) if args.only else None

    def transcribe(wav: Path, lang: str) -> tuple[str, str]:
        cmd = [args.mynah_asr, "transcribe", "-m", args.model, "-i", str(wav), "--lang", lang]
        if args.quant:
            cmd += ["--quant", args.quant]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        m = re.search(r"lang=([\w-]+)", proc.stderr)
        return proc.stdout.strip(), (m.group(1) if m else "?")

    # Primary criterion: CER with an EXPLICIT language (does ASR work for it?).
    # Secondary/informational: language detection with "auto" (on short clips the
    # model may fail to detect the language and emit nothing: known behaviour).
    # regional variants: same samples as the base language, the variant's prompt
    jobs = dict(sorted(manifest.items()))
    if has_prompt:   # the variants differ only by prompt: without one they are duplicates
        for variant, base in VARIANTS.items():
            if base in manifest:
                jobs[variant] = manifest[base]
    if only:
        missing = only - set(jobs)
        if missing:
            print(f"WARNING: requested locales without samples: {sorted(missing)}")
        jobs = {k: v for k, v in jobs.items() if k in only}

    n_lang_ok = n_lang_fail = 0
    failures = []
    print(f"{'locale':8} {'sample':>6} {'CER ok':>7} {'mean CER':>10} {'auto-lang':>10}  result")
    for locale, entries in jobs.items():
        cers, auto_hits = [], 0
        for e in entries:
            wav = ROOT / "tests/audio/langs" / e["wav"]
            hyp, _ = transcribe(wav, locale if has_prompt else "auto")
            c = cer(e["text"], hyp)
            cers.append(c)
            if has_prompt:   # separate detection only when a prompt exists
                _, detected = transcribe(wav, "auto")
                if detected.split("-")[0] == locale.split("-")[0]:
                    auto_hits += 1
            if c > args.cer_max:
                sample_id = e.get("tatoeba_audio_id") or e.get("fleurs_id") or "?"
                failures.append(f"  {locale} [{sample_id}] cer={c:.2f}\n"
                                f"    ref: {e['text']}\n    hyp: {hyp}")
        n_ok = sum(1 for c in cers if c <= args.cer_max)
        avg = sum(cers) / len(cers)
        ok = n_ok * 2 > len(cers)   # a majority of the samples below threshold
        if ok:
            result = "OK"
            n_lang_ok += 1
        elif locale in ADAPTATION_TIER and not only:
            result = "WEAK (adaptation tier: expected)"
        else:
            result = "FAIL"
            n_lang_fail += 1
        auto_col = f"{auto_hits}/{len(entries)}" if has_prompt else "n/a"
        print(f"{locale:8} {len(entries):>6} {n_ok}/{len(cers):>5} {avg:>10.3f} "
              f"{auto_col:>10}  {result}")

    print(f"\n[{args.quant or 'f32'}] {n_lang_ok} languages OK, {n_lang_fail} FAIL "
          f"(CER threshold {args.cer_max}, criterion: majority)")
    if failures:
        print("\nSamples above threshold, in detail:")
        print("\n".join(failures))
    sys.exit(0 if n_lang_fail == 0 else 1)


if __name__ == "__main__":
    main()
