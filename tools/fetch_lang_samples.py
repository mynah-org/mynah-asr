#!/usr/bin/env python3
"""Download real per-language audio samples from Tatoeba (short CC-audio sentences).

Per ogni lingua supportata da Nemotron 3.5 prende fino a N frasi con audio,
le converte a WAV 16 kHz mono (ffmpeg) e scrive un manifest con testo di
riferimento e attribuzione. I sample NON vanno committati (solo fixture locali,
licenze audio Tatoeba variabili): tests/audio/langs/ è in .gitignore.

Uso: uv run python fetch_lang_samples.py [n_per_lingua=3]
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "audio" / "langs"

# Nemotron locale -> Tatoeba ISO 639-3 code (regional variants share the
# language: en-GB is tested through eng, and so on)
LOCALES = {
    "en-US": "eng", "es-ES": "spa", "fr-FR": "fra", "it-IT": "ita", "pt-BR": "por",
    "nl-NL": "nld", "de-DE": "deu", "tr-TR": "tur", "ru-RU": "rus", "ar-AR": "ara",
    "hi-IN": "hin", "ja-JP": "jpn", "ko-KR": "kor", "vi-VN": "vie", "uk-UA": "ukr",
    "pl-PL": "pol", "sv-SE": "swe", "cs-CZ": "ces", "nb-NO": "nob", "da-DK": "dan",
    "bg-BG": "bul", "fi-FI": "fin", "hr-HR": "hrv", "sk-SK": "slk", "zh-CN": "cmn",
    "hu-HU": "hun", "ro-RO": "ron", "et-EE": "est", "el-GR": "ell", "lt-LT": "lit",
    "lv-LV": "lav", "mt-MT": "mlt", "sl-SI": "slv", "he-IL": "heb", "th-TH": "tha",
    "nn-NO": "nno",
}

API = "https://api.tatoeba.org/unstable/audios?lang={code}&limit=80"
AUDIO = "https://tatoeba.org/en/audio/download/{aid}"

# Fallback for languages without audio on Tatoeba: FLEURS (CC-BY-4.0), validation
# split through the datasets-server rows API (the test split exceeds the scan limit).
FLEURS = {
    "hi-IN": "hi_in", "ko-KR": "ko_kr", "vi-VN": "vi_vn", "nb-NO": "nb_no",
    "da-DK": "da_dk", "bg-BG": "bg_bg", "hr-HR": "hr_hr", "sk-SK": "sk_sk",
    "et-EE": "et_ee", "el-GR": "el_gr", "lt-LT": "lt_lt", "lv-LV": "lv_lv",
    "mt-MT": "mt_mt", "sl-SI": "sl_si",
}
FLEURS_API = ("https://datasets-server.huggingface.co/rows?dataset=google/fleurs"
              "&config={cfg}&split=validation&offset=0&length={n}")


def fetch_fleurs(locale: str, cfg: str, n: int, lang_dir: Path) -> list[dict]:
    entries = []
    rows = []
    for attempt in range(4):                # il datasets-server dà 5xx transitori
        try:
            rows = fetch_json(FLEURS_API.format(cfg=cfg, n=n * 2)).get("rows", [])
            break
        except Exception as e:
            if attempt == 3:
                print(f"{locale}: FLEURS API error ({e})")
                return entries
            time.sleep(3.0 * (attempt + 1))
    for r in rows:
        if len(entries) >= n:
            break
        row = r.get("row", {})
        text = row.get("raw_transcription", "")
        srcs = row.get("audio") or []
        if not text or not srcs:
            continue
        rid = row.get("id", len(entries))
        wav = lang_dir / f"fleurs_{rid}.wav"
        if not wav.exists():
            tmp = lang_dir / f"fleurs_{rid}.dl"
            try:
                urllib.request.urlretrieve(srcs[0]["src"], tmp)
                subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-i", str(tmp),
                                "-ar", "16000", "-ac", "1", str(wav)], check=True)
            except Exception:
                continue
            finally:
                tmp.unlink(missing_ok=True)
        entries.append({
            "wav": f"{locale}/fleurs_{rid}.wav",
            "text": text,
            "source": "google/fleurs validation (CC-BY-4.0)",
            "fleurs_id": rid,
        })
    return entries


def fetch_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "mynah-asr-tests"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main() -> None:
    n_per_lang = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, list[dict]] = {}
    report = []

    for locale, code in LOCALES.items():
        lang_dir = OUT_DIR / locale
        lang_dir.mkdir(exist_ok=True)
        got = 0
        try:
            data = fetch_json(API.format(code=code)).get("data", [])
        except Exception:
            data = []   # nessun continue: lascia lavorare il fallback FLEURS

        # CJK/Thai: frasi brevi in caratteri ma dense — soglia ridotta
        min_len = 4 if code in {"jpn", "cmn", "kor", "tha"} else 12
        entries = []
        for a in data:
            if got >= n_per_lang:
                break
            sent = a.get("sentence") or {}
            text = sent.get("text", "")
            if len(text) < min_len or sent.get("lang") != code:
                continue
            aid = a["id"]
            wav = lang_dir / f"{aid}.wav"
            if not wav.exists():
                mp3 = lang_dir / f"{aid}.mp3"
                try:
                    urllib.request.urlretrieve(AUDIO.format(aid=aid), mp3)
                    subprocess.run(
                        ["ffmpeg", "-v", "quiet", "-y", "-i", str(mp3), "-ar", "16000",
                         "-ac", "1", str(wav)],
                        check=True,
                    )
                except Exception:
                    continue
                finally:
                    mp3.unlink(missing_ok=True)
            entries.append({
                "wav": f"{locale}/{aid}.wav",
                "text": text,
                "tatoeba_audio_id": aid,
                "tatoeba_sentence_id": sent.get("id"),
                "audio_license": a.get("license"),
                "author": a.get("author"),
                "attribution_url": a.get("attribution_url"),
            })
            got += 1
            time.sleep(0.2)

        if not entries and locale in FLEURS:
            entries = fetch_fleurs(locale, FLEURS[locale], n_per_lang, lang_dir)
            got = len(entries)
        if entries:
            manifest[locale] = entries
        report.append(f"{locale}: {got} sample")

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1))
    print("\n".join(report))
    total = sum(len(v) for v in manifest.values())
    print(f"\nTotale: {total} sample in {len(manifest)} lingue -> {OUT_DIR}")


if __name__ == "__main__":
    main()
