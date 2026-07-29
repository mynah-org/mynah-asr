#!/usr/bin/env python3
"""QUALITY test on real audio (samples/ — FLEURS, CC-BY 4.0).

Per ogni modello presente e ogni sample applicabile:
- ASR: CER normalizzato vs trascrizione di riferimento (soglia --cer-max)
- Speech translation (modelli AED/Canary): X->en vs riferimento inglese
  PARALLELO (stessa frase FLEURS) + en->de vs riferimento tedesco, con
  word-overlap sulle parole di contenuto (soglia --overlap-min: le
  traduzioni legittime variano, il CER sarebbe ingiusto)
- Backend: cpu e, su macOS, metal (--backend per restringere; "cuda" per
  la validazione Linux futura)

Exit: 0 ok, 1 fail, 77 skip (samples o modelli assenti).
Uso: uv run python -m eval.test_samples [--models a,b] [--backend cpu]
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

from eval.test_langs import cer, normalize

ROOT = Path(__file__).resolve().parent.parent.parent

# modello -> (lingue ASR coperte dai sample, formato tag lingua)
# nemotron: 40 locales (11 with samples here); v3: 25 EU languages (no ja)
MODELS = {
    "nemotron-3.5-asr-streaming-0.6b": {
        "langs": ["it", "en", "de", "es", "fr", "pt", "nl", "pl", "ru", "uk", "ja"],
        "tag": "locale"},
    "parakeet-tdt-0.6b-v3": {
        "langs": ["it", "en", "de", "es", "fr", "pt", "nl", "pl", "ru", "uk"],
        "tag": "auto"},
    "parakeet-tdt_ctc-110m": {"langs": ["en"], "tag": "auto"},
    "canary-180m-flash": {"langs": ["en", "de", "es", "fr"], "tag": "short"},
    "canary-1b-v2": {"langs": ["it", "en", "de", "es", "fr", "pt", "nl", "pl", "ru", "uk"],
                     "tag": "short"},
}
LOCALE = {"it": "it-IT", "en": "en-US", "de": "de-DE", "es": "es-ES", "fr": "fr-FR",
          "pt": "pt-BR", "nl": "nl-NL", "pl": "pl-PL", "ru": "ru-RU", "uk": "uk-UA",
          "ja": "ja-JP"}


def word_overlap(ref: str, hyp: str, min_len: int = 4) -> float:
    """Recall of the reference content words inside the hypothesis, with matching
    per prefisso (5 char): le traduzioni legittime variano la morfologia
    (relative/relatively, inaccessibility/inaccessible) e il match esatto le
    boccerebbe (visto su FLEURS 1534 es>en: parafrasi corretta, overlap 0.18)."""
    rw = {w for w in normalize(ref).split() if len(w) >= min_len}
    hw = {w for w in normalize(hyp).split() if len(w) >= min_len}
    if not rw:
        return 1.0
    hits = sum(1 for r in rw
               if any(r[:5] == h[:5] or r.startswith(h) or h.startswith(r) for h in hw))
    return hits / len(rw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cer-max", type=float, default=0.20)
    ap.add_argument("--overlap-min", type=float, default=0.35)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--backend", default=None, help="cpu|metal|cuda (default: cpu+metal su macOS)")
    ap.add_argument("--mynah-asr", dest="mynah_asr", default=str(ROOT / "mynah-asr"))
    args = ap.parse_args()

    manifest_path = ROOT / "samples/manifest.json"
    if not manifest_path.exists():
        print("SKIP: samples/ assente (tools/fetch_fleurs_samples.py)")
        sys.exit(77)
    samples = json.loads(manifest_path.read_text())["samples"]
    by_lang: dict[str, list[dict]] = {}
    long_samples = []
    for s in samples:
        if s.get("long"):
            long_samples.append(s)
        else:
            by_lang.setdefault(s["lang"], []).append(s)

    models = [m for m in args.models.split(",")
              if (ROOT / "models" / m / "mynah.json").exists()]
    if not models:
        print("SKIP: nessun modello presente")
        sys.exit(77)
    backends = [args.backend] if args.backend else (
        ["cpu", "metal"] if platform.system() == "Darwin" else ["cpu"])

    def run(model: str, wav: str, lang: str, backend: str, target: str | None = None) -> str:
        cmd = [args.mynah_asr, "transcribe", "-m", str(ROOT / "models" / model),
               "-i", str(ROOT / "samples" / wav), "--lang", lang, "--backend", backend]
        if target:
            cmd += ["--target-lang", target]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return p.stdout.strip()

    fails = 0
    for model in models:
        cfg = MODELS.get(model, {"langs": ["en"], "tag": "auto"})
        is_aed = "canary" in model
        for backend in backends:
            for lang in cfg["langs"]:
                for s in by_lang.get(lang, []):
                    tag = {"locale": LOCALE[lang], "auto": "auto", "short": lang}[cfg["tag"]]
                    hyp = run(model, s["file"], tag, backend)
                    c = cer(s["text"], hyp)
                    ok = c <= args.cer_max
                    print(f"{'OK ' if ok else 'FAIL'} asr  {model:32s} {backend:5s} "
                          f"{s['file']:22s} CER {c:.3f}")
                    if not ok:
                        print(f"     ref: {s['text']}\n     hyp: {hyp}")
                        fails += 1
            if is_aed:
                # traduzione X->en (riferimento = frase inglese parallela)
                for lang in ("de", "es", "fr"):
                    for s in by_lang.get(lang, []):
                        hyp = run(model, s["file"], lang, backend, target="en")
                        ov = word_overlap(s["en_ref"], hyp)
                        ok = ov >= args.overlap_min
                        print(f"{'OK ' if ok else 'FAIL'} trx  {model:32s} {backend:5s} "
                              f"{s['file']:22s} {lang}>en overlap {ov:.2f}")
                        if not ok:
                            print(f"     ref: {s['en_ref']}\n     hyp: {hyp}")
                            fails += 1
                # en->de (riferimento = frase tedesca parallela)
                de_by_id = {x["fleurs_id"]: x for x in by_lang.get("de", [])}
                for s in by_lang.get("en", []):
                    ref = de_by_id.get(s["fleurs_id"])
                    if not ref:
                        continue
                    hyp = run(model, s["file"], "en", backend, target="de")
                    ov = word_overlap(ref["text"], hyp)
                    ok = ov >= args.overlap_min
                    print(f"{'OK ' if ok else 'FAIL'} trx  {model:32s} {backend:5s} "
                          f"{s['file']:22s} en>de overlap {ov:.2f}")
                    if not ok:
                        print(f"     ref: {ref['text']}\n     hyp: {hyp}")
                        fails += 1
    # clip lungo: segmentazione su silenzio, timestamp monotoni, streaming reale
    # (cpu only: the metal paths were exercised above; here the features matter)
    def rtf_of(stderr: str) -> str:
        m = re.search(r"RTF ([0-9.]+)", stderr)
        return m.group(1) if m else "?"

    for s in long_samples:
        wav = str(ROOT / "samples" / s["file"])
        if s["file"].startswith("long/"):
            # long transcribe / long translate: model-aware segmentation
            # defaults (measures out-of-the-box quality) + RTF
            if s["lang"] == "en" and "parakeet-tdt-0.6b-v3" in models:
                p = subprocess.run([args.mynah_asr, "transcribe", "-m",
                                    str(ROOT / "models/parakeet-tdt-0.6b-v3"),
                                    "-i", wav], capture_output=True, text=True,
                                   timeout=1200)
                c = cer(s["text"], p.stdout.strip())
                ok = c <= args.cer_max
                print(f"{'OK ' if ok else 'FAIL'} long parakeet-tdt-0.6b-v3 "
                      f"{s['file']} ({s['duration_sec']}s) CER {c:.3f} "
                      f"RTF {rtf_of(p.stderr)}")
                fails += 0 if ok else 1
            if s.get("en_ref") and "canary-180m-flash" in models:
                p = subprocess.run([args.mynah_asr, "transcribe", "-m",
                                    str(ROOT / "models/canary-180m-flash"),
                                    "-i", wav, "--lang", s["lang"],
                                    "--target-lang", "en"], capture_output=True,
                                   text=True, timeout=1200)
                ov = word_overlap(s["en_ref"], p.stdout.strip())
                ok = ov >= args.overlap_min
                print(f"{'OK ' if ok else 'FAIL'} long canary-180m-flash "
                      f"{s['file']} ({s['duration_sec']}s) {s['lang']}>en "
                      f"overlap {ov:.2f} RTF {rtf_of(p.stderr)}")
                if not ok:
                    print(f"     hyp: {p.stdout.strip()[:200]}")
                fails += 0 if ok else 1
            continue
        if "parakeet-tdt-0.6b-v3" in models:
            p = subprocess.run([args.mynah_asr, "transcribe", "-m",
                                str(ROOT / "models/parakeet-tdt-0.6b-v3"), "-i", wav,
                                "--segment-sec", "30"],
                               capture_output=True, text=True, timeout=600)
            c = cer(s["text"], p.stdout.strip())
            ok = c <= args.cer_max
            print(f"{'OK ' if ok else 'FAIL'} seg  parakeet-tdt-0.6b-v3 "
                  f"{s['file']} ({s['duration_sec']}s, segmenti da 30s) CER {c:.3f}")
            fails += 0 if ok else 1

            p = subprocess.run([args.mynah_asr, "transcribe", "-m",
                                str(ROOT / "models/parakeet-tdt-0.6b-v3"), "-i", wav,
                                "--timestamps"],
                               capture_output=True, text=True, timeout=600)
            rows = [ln.split() for ln in p.stdout.splitlines() if len(ln.split()) >= 3]
            t0s = [float(r[0]) for r in rows]
            ok = (len(rows) > 50 and t0s == sorted(t0s)
                  and float(rows[-1][1]) <= s["duration_sec"] + 1.0)
            print(f"{'OK ' if ok else 'FAIL'} ts   parakeet-tdt-0.6b-v3 "
                  f"{s['file']} {len(rows)} parole, monotoni, entro durata")
            fails += 0 if ok else 1

        if "nemotron-3.5-asr-streaming-0.6b" in models:
            import soundfile as sf
            audio, _sr = sf.read(ROOT / "samples" / s["file"], dtype="int16")
            p = subprocess.run([args.mynah_asr, "stream", "-m",
                                str(ROOT / "models/nemotron-3.5-asr-streaming-0.6b"),
                                "--lang", "en-US"],
                               input=audio.tobytes(), capture_output=True, timeout=600)
            c = cer(s["text"], p.stdout.decode().strip())
            ok = c <= args.cer_max
            print(f"{'OK ' if ok else 'FAIL'} strm nemotron-3.5 (cache-aware) "
                  f"{s['file']} CER {c:.3f}")
            fails += 0 if ok else 1

    print(f"\n{'FAIL' if fails else 'OK'}: {fails} problemi su {len(models)} modelli, "
          f"backend {backends}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
