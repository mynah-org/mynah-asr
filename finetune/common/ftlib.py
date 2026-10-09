"""Shared helpers for the Canary 180M Italian fine-tuning kit.

Pure-Python parts (manifests, text normalisation, WER/CER with S/D/I) import
nothing heavy, so they can be unit-checked on a laptop. The decode loop imports
torch lazily.

Normalisation follows the HF Open ASR Leaderboard multilingual path
(`normalizer/data_utils.py::MultilingualNormalizer(remove_diacritics=False)`
called with `lang=`), which is Whisper's `BasicMultilingualTextNormalizer`
(MIT, OpenAI; Apache-2.0 copy in huggingface/open_asr_leaderboard) plus a
num2words pass. The basic part is reproduced below so the Italian score does
not depend on a network fetch; `setup.sh` fetches the leaderboard's own
`normalizer.py` (pinned commit) for the English normaliser, and
`selfcheck_normalizer()` compares the two on sample strings.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

try:
    import regex as _re_u  # what the leaderboard uses for [^\w\s]
except ImportError:  # pragma: no cover - laptop fallback
    _re_u = re

FT = Path(os.environ.get("FT_ROOT", "/root/ft"))
SR = 16000
SPECIAL_RE = re.compile(r"<\|[^|]*\|>|<pad>|<unk>|<s>|</s>")

# --------------------------------------------------------------------------
# manifests
# --------------------------------------------------------------------------


def read_manifest(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def canary_row(audio_filepath, duration, text, lang, pnc, **extra):
    """One NeMo/lhotse manifest row for a canary2 model (Canary 180M flash).

    Field names checked against NeMo v3.0.0:
    - `audio_filepath`, `duration`, `text` (our text_field), `sampling_rate`
      (lets lhotse skip an audio probe):
      nemo/collections/common/data/lhotse/nemo_adapters.py (LazyNeMoIterator);
      every other key lands in `cut.custom`.
    - `source_lang`, `target_lang`: REQUIRED by
      nemo/collections/common/prompts/canary2.py::canary2(); `target_lang` is
      also the lang_field that picks the sub-tokenizer for the answer.
    - `pnc`: optional ("yes"/"no"; default <|pnc|>). The other canary2 slots
      (`itn`, `timestamp`, `diarize`, `emotion`, `decodercontext`) are left
      out ON PURPOSE so canary2() applies its defaults (<|noitn|>,
      <|notimestamp|>, <|nodiarize|>, <|emo:undefined|>, ""); ftlib.prompt_ids
      uses the same values at decode time.
    - `taskname`: the canary(1) slot; canary2 ignores it, kept for tools that
      still read it.
    """
    row = {
        "audio_filepath": str(audio_filepath),
        "duration": round(float(duration), 3),
        "sampling_rate": SR,
        "text": text,
        "source_lang": lang,
        "target_lang": lang,
        "pnc": pnc,
        "taskname": "asr",
    }
    row.update(extra)
    return row


def mix_replay(it_rows, replay_rows, ratio, seed):
    """Training rows where ~`ratio` of the audio seconds are replay rows.

    Replay seconds target = ratio / (1 - ratio) x IT seconds; replay rows are
    taken in whole shuffled passes over the pool (repeating it if the target
    exceeds the pool) and the last pass is cut at the target. The result is
    shuffled as a whole (deterministic `seed`): lhotse's shuffle buffer is far
    smaller than an epoch, so an IT-then-replay file would arrive in blocks.
    Returns (rows, info)."""
    import random

    assert 0.0 < ratio < 1.0, ratio
    it_s = sum(float(r["duration"]) for r in it_rows)
    pool_s = sum(float(r["duration"]) for r in replay_rows)
    assert pool_s > 0, "empty replay pool"
    target = ratio / (1.0 - ratio) * it_s
    rng = random.Random(seed)
    picked, acc, passes = [], 0.0, 0
    while acc < target:
        order = list(replay_rows)
        rng.shuffle(order)
        passes += 1
        for r in order:
            if acc >= target:
                break
            picked.append(r)
            acc += float(r["duration"])
    rows = list(it_rows) + picked
    rng.shuffle(rows)
    per_lang = {}
    for r in picked:
        per_lang[r["target_lang"]] = per_lang.get(r["target_lang"], 0.0) + float(r["duration"])
    info = {"ratio": ratio, "it_s": round(it_s, 1), "replay_s": round(acc, 1), "replay_pool_s": round(pool_s, 1),
            "replay_share": round(acc / (acc + it_s), 4), "replay_rows": len(picked), "pool_passes": passes,
            "replay_s_per_lang": {k: round(v, 1) for k, v in sorted(per_lang.items())}}
    return rows, info


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------


def _remove_symbols_keep_marks(s):
    return "".join(" " if unicodedata.category(c)[0] in "SP" else c for c in unicodedata.normalize("NFKC", s))


def basic_multilingual(s):
    """Whisper BasicMultilingualTextNormalizer(remove_diacritics=False)."""
    s = s.lower()
    s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)
    s = re.sub(r"\(([^)]+?)\)", "", s)
    s = _remove_symbols_keep_marks(s).lower()
    s = _re_u.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _numbers(s, lang):
    s = re.sub(r"(\d)\s+(\d{3})\b", r"\1\2", s)
    try:
        import num2words
    except ImportError:
        return s

    def rep(m):
        try:
            return num2words.num2words(int(m.group()), lang=lang)
        except Exception:
            return m.group()

    return re.sub(r"\d+", rep, s)


def normalize_multilingual(s, lang):
    """open_asr_leaderboard MultilingualNormalizer(remove_diacritics=False)(s, lang=lang)
    (the FILLER_WORDS table is empty upstream, so there is no filler pass)."""
    return _numbers(basic_multilingual(s), lang)


_EN = None


def _english():
    global _EN
    if _EN is None:
        vend = FT / "vendor"
        if (vend / "oal_norm" / "normalizer.py").exists():
            sys.path.insert(0, str(vend))
            from oal_norm.normalizer import EnglishTextNormalizer  # type: ignore

            _EN = ("leaderboard EnglishTextNormalizer", EnglishTextNormalizer())
        else:
            _EN = ("FALLBACK basic multilingual (vendor/oal_norm missing)", None)
    return _EN


def normalize(s, lang):
    if lang == "en":
        name, en = _english()
        if en is not None:
            return re.sub(r"\s+", " ", en(s)).strip()
    return normalize_multilingual(s, lang)


def normalizer_name(lang):
    if lang == "en":
        return _english()[0]
    return "leaderboard MultilingualNormalizer(remove_diacritics=False), lang=" + lang


def selfcheck_normalizer():
    """Compare the local multilingual normaliser with the fetched leaderboard one."""
    vend = FT / "vendor"
    if not (vend / "oal_norm" / "normalizer.py").exists():
        return "skipped (vendor/oal_norm missing)"
    sys.path.insert(0, str(vend))
    from oal_norm.normalizer import BasicMultilingualTextNormalizer  # type: ignore

    ref = BasicMultilingualTextNormalizer(remove_diacritics=False)
    samples = [
        "L'incidente è avvenuto in alta montagna, e si ritiene sia stato causato da fuoco nemico.",
        "Perché? «Così» — disse: 3 000 euro (circa) [rumore] nel 1999!",
        "Ëlite d'Annunzio... ÀÈÌÒÙ àèìòù",
    ]
    bad = [s for s in samples if ref(s) != basic_multilingual(s)]
    return "ok" if not bad else f"MISMATCH on {bad}"


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def align_counts(ref, hyp):
    """Levenshtein over two token lists -> (S, D, I). Ties prefer S, then D, then I."""
    n, m = len(ref), len(hyp)
    prev = [(j, 0, 0, j) for j in range(m + 1)]  # (cost, S, D, I)
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)] + [None] * m
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur[j] = prev[j - 1]
                continue
            c_s, c_d, c_i = prev[j - 1], prev[j], cur[j - 1]
            best = min(
                (c_s[0] + 1, 0, (c_s[0] + 1, c_s[1] + 1, c_s[2], c_s[3])),
                (c_d[0] + 1, 1, (c_d[0] + 1, c_d[1], c_d[2] + 1, c_d[3])),
                (c_i[0] + 1, 2, (c_i[0] + 1, c_i[1], c_i[2], c_i[3] + 1)),
            )
            cur[j] = best[2]
        prev = cur
    _, s, d, ins = prev[m]
    return s, d, ins


def score(pairs, lang):
    """pairs: list of (ref_raw, hyp_raw). Returns corpus WER/CER with S/D/I."""
    w = {"S": 0, "D": 0, "I": 0, "N": 0}
    c = {"S": 0, "D": 0, "I": 0, "N": 0}
    empty_hyp = 0
    per_utt = []
    for ref_raw, hyp_raw in pairs:
        r, h = normalize(ref_raw, lang), normalize(hyp_raw, lang)
        rw, hw = r.split(), h.split()
        s, d, i = align_counts(rw, hw)
        w["S"] += s
        w["D"] += d
        w["I"] += i
        w["N"] += len(rw)
        cs, cd, ci = align_counts(list(r), list(h))
        c["S"] += cs
        c["D"] += cd
        c["I"] += ci
        c["N"] += len(r)
        empty_hyp += not hw
        per_utt.append({"ref_norm": r, "hyp_norm": h, "errs": s + d + i, "nref": len(rw)})

    def rate(x):
        return round(100.0 * (x["S"] + x["D"] + x["I"]) / max(1, x["N"]), 3)

    out = {
        "n_utts": len(pairs),
        "wer": rate(w),
        "cer": rate(c),
        "words": w,
        "chars": c,
        "empty_hyp": empty_hyp,
        "normalizer": normalizer_name(lang),
    }
    return out, per_utt


# --------------------------------------------------------------------------
# NeMo decode (bypasses model.transcribe on purpose)
# --------------------------------------------------------------------------
#
# Why not transcribe(): its lhotse path encodes the assistant turn with the
# target language's sub-tokenizer, and CanaryTokenizer raises "Unsupported
# language" for a language without a text sub-tokenizer -- so the zero-shot
# probe (<|it|> on the untouched model) cannot go through it. Building the
# canary2 *user* turn ourselves only needs the special-token tokenizer, which
# already contains <|it|>. Same code path for every checkpoint, so numbers are
# comparable across probe, surgery check and fine-tuned runs.


def load_model(path, device="cuda"):
    from nemo.collections.asr.models import EncDecMultiTaskModel

    m = EncDecMultiTaskModel.restore_from(str(path), map_location="cpu")
    m = m.to(device)
    m.eval()
    return m


def prompt_ids(model, src, tgt, pnc="no"):
    # same values as the training-time defaults of canary2() (see canary_row)
    slots = {
        "decodercontext": "",
        "emotion": "<|emo:undefined|>",
        "source_lang": src,
        "target_lang": tgt,
        "pnc": pnc,
        "itn": "no",
        "timestamp": "no",
        "diarize": "no",
    }
    # restrict to the slots the model's prompt format actually has
    wanted = set(model.prompt.get_slots("user"))
    slots = {k: v for k, v in slots.items() if k in wanted}
    return model.prompt.encode_dialog(turns=[{"role": "user", "slots": slots}])["context_ids"]


def relevel(a, peak_dbfs):
    """Scale a clip so its peak sits at `peak_dbfs`, then re-quantise to the
    PCM16 grid (what a real low-level 16-bit recording would carry)."""
    import numpy as np

    peak = float(np.max(np.abs(a))) if len(a) else 0.0
    if peak <= 0:
        return a
    a = a * (10.0 ** (peak_dbfs / 20.0) / peak)
    return (np.round(np.clip(a, -1.0, 1.0) * 32767.0) / 32767.0).astype(np.float32)


def peak_dbfs(a):
    import numpy as np

    p = float(np.max(np.abs(a))) if len(a) else 0.0
    return round(float(20.0 * np.log10(p)), 2) if p > 0 else -120.0


def _load_audio(path, level=None):
    import numpy as np
    import soundfile as sf

    a, sr = sf.read(path, dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if sr != SR:
        import torch
        import torchaudio.functional as AF

        a = AF.resample(torch.from_numpy(a), sr, SR).numpy()
    if level is not None:
        a = relevel(a, level)
    return np.ascontiguousarray(a)


def decode(model, rows, src, tgt, pnc="no", batch_size=32, bf16=False, min_dur=1.0, verbose=True, level=None):
    """Greedy/beam(1) decode with the model's own decoding config.

    rows need `audio_filepath`; returns hypotheses in the order of `rows`.
    `level` (peak dBFS, e.g. -40) re-levels every clip before decoding.
    Clips shorter than `min_dur` are zero-padded (both sides) to `min_dur`,
    like the Canary card's evaluation and transcribe()'s pad_min_duration=1.0.
    """
    import numpy as np
    import torch

    dev = next(model.parameters()).device
    p = prompt_ids(model, src, tgt, pnc)
    order = sorted(range(len(rows)), key=lambda i: -float(rows[i].get("duration", 0)))
    hyps = [None] * len(rows)
    t0 = time.time()
    was_training = model.training
    model.eval()
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        wavs = [_load_audio(rows[i]["audio_filepath"], level) for i in idx]
        minlen = int(min_dur * SR)
        for k, w in enumerate(wavs):
            if len(w) < minlen:
                pad = minlen - len(w)
                wavs[k] = np.pad(w, (pad // 2, pad - pad // 2))
        lens = torch.tensor([len(w) for w in wavs], dtype=torch.long, device=dev)
        x = torch.zeros(len(wavs), int(lens.max()), dtype=torch.float32)
        for k, w in enumerate(wavs):
            x[k, : len(w)] = torch.from_numpy(w)
        x = x.to(dev)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            _, _, enc, enc_mask = model.forward(input_signal=x, input_signal_length=lens)
            dec_in = p.unsqueeze(0).repeat(len(idx), 1).to(dev)
            out = model.decoding.decode_predictions_tensor(
                encoder_hidden_states=enc, encoder_input_mask=enc_mask, decoder_input_ids=dec_in, return_hypotheses=False
            )
        for i, h in zip(idx, out):
            h = h[0] if isinstance(h, list) else h
            txt = h.text if hasattr(h, "text") else str(h)
            hyps[i] = re.sub(r"\s+", " ", SPECIAL_RE.sub(" ", txt)).strip()
        if verbose and (b // batch_size) % 20 == 0:
            print(f"  decode {min(b + batch_size, len(order))}/{len(order)} {time.time() - t0:.0f}s", flush=True)
    if was_training:
        model.train()
    return hyps


def eval_sets(model, sets, lang, pnc="no", batch_size=32, bf16=False, limit=None, dump_dir=None, level=None):
    """sets: {name: manifest_path}. Returns {name: score dict, '_pooled': ...}.
    `level`: None = audio as recorded; a number = every clip re-levelled to that peak dBFS."""
    res, pooled = {}, []
    t0 = time.time()
    for name, man in sets.items():
        rows = read_manifest(man)
        if limit:
            rows = rows[:limit]
        audio_s = sum(float(r["duration"]) for r in rows)
        t1 = time.time()
        hyps = decode(model, rows, lang, lang, pnc=pnc, batch_size=batch_size, bf16=bf16, level=level)
        dt = time.time() - t1
        pairs = [(r.get("ref", r["text"]), h) for r, h in zip(rows, hyps)]
        sc, per = score(pairs, lang)
        sc.update({"level_peak_dbfs": level, "manifest": str(man), "audio_h": round(audio_s / 3600, 4), "decode_wall_s": round(dt, 1),
                   "rtfx": round(audio_s / max(dt, 1e-9), 1)})
        res[name] = sc
        pooled += pairs
        if dump_dir:
            sfx = "" if level is None else f"_peak{int(level)}"
            write_manifest(Path(dump_dir) / f"hyp_{name}{sfx}.jsonl",
                           [{"audio_filepath": r["audio_filepath"], "ref": a, "hyp": h, **u}
                            for r, (a, h), u in zip(rows, pairs, per)])
        print(f"  [{name}{'' if level is None else f' @peak {level:g} dBFS'}] n={sc['n_utts']} WER {sc['wer']:.2f} CER {sc['cer']:.2f} "
              f"S/D/I {sc['words']['S']}/{sc['words']['D']}/{sc['words']['I']} empty={sc['empty_hyp']} "
              f"({dt:.0f}s, RTFx {sc['rtfx']})", flush=True)
    if len(sets) > 1:
        res["_pooled"], _ = score(pooled, lang)
    res["_wall_s"] = round(time.time() - t0, 1)
    return res


def parse_levels(s):
    return [float(x) for x in str(s).split(",") if x.strip() not in ("", "native")]


def level_sweep(model, sets, lang, levels, batch_size=32, dump_dir=None):
    """WER per set at each fixed peak level -> {"-3": {set: wer, ...}, ...}."""
    out = {}
    for lv in levels:
        r = eval_sets(model, sets, lang, pnc="no", batch_size=batch_size, dump_dir=dump_dir, level=lv)
        out[f"{lv:g}"] = {k: {"wer": v["wer"], "cer": v["cer"], "empty_hyp": v["empty_hyp"]}
                          for k, v in r.items() if isinstance(v, dict)}
    return out


def preprocessor_info(model):
    """Level handling of the front end: `normalize` (per_feature / NA / ...),
    dither, log guard. per_feature normalisation removes most of a pure gain
    change; NA (no normalisation) does not."""
    p = model.cfg.preprocessor
    return {k: (str(p.get(k)) if p.get(k) is not None else None)
            for k in ("_target_", "normalize", "dither", "log", "log_zero_guard_type", "log_zero_guard_value",
                      "features", "window_stride")}


def make_gain_aug(lo_db=-30.0, hi_db=6.0, prob=0.8, seed=0, audio_attr="audio"):
    """Lightning callback: random per-utterance gain on the training batch.

    NeMo v3.0.0's lhotse loader has no plain gain/volume option (it has noise,
    speed, RIR, low-pass, compression and clipping-with-gain), so the gain is
    applied to the already-padded batch on the device, just before
    training_step (padding stays zero), then clipped to [-1, 1] and
    re-quantised to the PCM16 grid like a real 16-bit capture. Logged:
    applied fraction and mean/min/max dB. `audio_attr` names the batch field
    holding the [B, T] audio ("audio" for the canary2 batch, "audio_signal" for
    NeMo's AudioToTextEOUBatch).
    """
    import lightning.pytorch as pl
    import torch

    class GainAug(pl.Callback):
        def __init__(self):
            self.g = torch.Generator().manual_seed(seed)
            self.stats = {"lo_db": lo_db, "hi_db": hi_db, "prob": prob, "n": 0, "applied": 0, "sum_db": 0.0,
                          "min_db": None, "max_db": None}

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if batch is None:
                return
            a = getattr(batch, audio_attr)
            n = a.shape[0]
            db = lo_db + (hi_db - lo_db) * torch.rand(n, generator=self.g)
            on = torch.rand(n, generator=self.g) < prob
            db = torch.where(on, db, torch.zeros_like(db))
            gain = (10.0 ** (db / 20.0)).to(a.device, a.dtype).unsqueeze(1)
            with torch.no_grad():
                a.mul_(gain).clamp_(-1.0, 1.0)
                a.copy_(torch.round(a * 32767.0) / 32767.0)
            st = self.stats
            st["n"] += n
            st["applied"] += int(on.sum())
            if on.any():
                d = db[on]
                st["sum_db"] += float(d.sum())
                st["min_db"] = float(d.min()) if st["min_db"] is None else min(st["min_db"], float(d.min()))
                st["max_db"] = float(d.max()) if st["max_db"] is None else max(st["max_db"], float(d.max()))

        def summary(self):
            st = dict(self.stats)
            st["mean_db_when_applied"] = round(st["sum_db"] / st["applied"], 2) if st["applied"] else None
            return st

    return GainAug()


def env_info():
    info = {"python": sys.version.split()[0]}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as e:  # pragma: no cover
        info["torch"] = f"unavailable: {e}"
    try:
        import nemo

        info["nemo"] = nemo.__version__
    except Exception:
        pass
    return info
