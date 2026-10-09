"""Energy trim of a clip to its voiced span (stand-in for a forced aligner).

Moved from the EOU kit (eou/train_eou.py): the NVIDIA EOU recipe wants single
utterances WITHOUT leading/trailing silence; rows get `offset`/`duration` of
the voiced part instead of re-written audio.
"""
from __future__ import annotations


def voiced_span_of(a, sr, thr_db=35.0, margin_s=0.10):
    """(offset_s, duration_s) of the frames (25 ms) whose RMS is within `thr_db`
    of the loudest frame, +- `margin_s`. `a` is a 1-D float array."""
    import numpy as np

    hop = int(0.025 * sr)
    n = len(a) // hop
    if n < 4:
        return 0.0, len(a) / sr
    fr = a[: n * hop].reshape(n, hop)
    db = 10 * np.log10(np.maximum((fr ** 2).mean(axis=1), 1e-12))
    on = np.nonzero(db > db.max() - thr_db)[0]
    s = max(0.0, on[0] * hop / sr - margin_s)
    e = min(len(a) / sr, (on[-1] + 1) * hop / sr + margin_s)
    return round(s, 3), round(e - s, 3)


def voiced_span(path, thr_db=35.0, margin_s=0.10):
    import soundfile as sf

    a, sr = sf.read(path, dtype="float32", always_2d=True)
    return voiced_span_of(a.mean(axis=1), sr, thr_db, margin_s)
