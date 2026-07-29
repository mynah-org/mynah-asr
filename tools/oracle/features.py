"""NeMo log-mel in pure numpy — an exact replica of the HF extractors
(reference/transformers-nemotron_asr_streaming/ e reference/transformers-parakeet/)
con torch.stft(center=True, pad_mode="constant") e finestra Hann simmetrica.

normalize: "NA" (Nemotron, nessuna normalizzazione) o "per_feature" (Parakeet:
media/std per bin sui frame validi, ddof=1, x = (x-mu)/(std+1e-5), come
ParakeetFeatureExtractor).
"""

from __future__ import annotations

import numpy as np

LOG_ZERO_GUARD = 2.0 ** -24


def log_mel(
    audio: np.ndarray,          # [S] float32 in [-1, 1], 16 kHz mono
    mel_fb: np.ndarray,         # [n_fft//2+1, n_mels] dal convertitore
    window: np.ndarray,         # [win_length] Hann simmetrica dal convertitore
    n_fft: int = 512,
    hop: int = 160,
    preemphasis: float = 0.97,
    normalize: str = "NA",
) -> tuple[np.ndarray, int]:
    """Returns (features [T, n_mels] float32, valid_len).

    T = 1 + S//hop (center=True); frames >= valid_len (= S//hop) are zeroed,
    come fa il feature extractor HF.
    """
    x = audio.astype(np.float64)
    S = x.shape[0]

    # preemphasis: y[0] = x[0], y[n] = x[n] - a*x[n-1]
    y = np.concatenate([x[:1], x[1:] - preemphasis * x[:-1]])

    # center pad (constant, zeros) of n_fft//2 per side
    pad = n_fft // 2
    y = np.pad(y, (pad, pad))

    # window: Hann of win_length, center-padded to n_fft (like torch.stft)
    win = np.zeros(n_fft, dtype=np.float64)
    off = (n_fft - window.shape[0]) // 2
    win[off:off + window.shape[0]] = window.astype(np.float64)

    n_frames = 1 + S // hop
    frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop][:n_frames]
    spec = np.fft.rfft(frames * win, n=n_fft, axis=-1)
    power = spec.real ** 2 + spec.imag ** 2            # [T, 257]

    mel = power @ mel_fb.astype(np.float64)             # [T, n_mels]
    feats = np.log(mel + LOG_ZERO_GUARD)

    valid = S // hop
    feats[valid:] = 0.0

    if normalize == "per_feature" and valid > 1:
        v = feats[:valid]
        mu = v.mean(axis=0)
        std = np.sqrt(((v - mu) ** 2).sum(axis=0) / (valid - 1))   # ddof=1
        feats[:valid] = (v - mu) / (std + 1e-5)

    return feats.astype(np.float32), valid
