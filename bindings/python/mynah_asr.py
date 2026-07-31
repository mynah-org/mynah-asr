"""Python bindings for libmynah_asr (ctypes, zero dependencies).

Prerequisite: `make shared` in the repo root (produces libmynah_asr.dylib/.so).

    from mynah_asr import MynahASR
    m = MynahASR("models/parakeet-tdt-0.6b-v3")
    print(m.transcribe("audio.wav"))
    text, words, lang = m.transcribe("audio.wav", timestamps=True)
    # translation (AED/Canary models):
    print(MynahASR("models/canary-180m-flash").transcribe("de.wav", lang="de>en"))
    # unknown source language: a detector in front of the translator
    lid = MynahASR("models/nemotron-3.5-asr-streaming-0.6b", "int8")
    canary = MynahASR("models/canary-1b-v2", "int8")
    print(canary.transcribe("unknown.wav", lang="auto>en", detector=lid))
"""

from __future__ import annotations

import array
import ctypes
import warnings
import wave
from pathlib import Path

QUANT = {"f32": 0, "int8": 1, "int4": 2}


def _find_lib() -> ctypes.CDLL:
    here = Path(__file__).resolve()
    candidates = []
    for base in (here.parent, here.parent.parent.parent, Path.cwd()):
        candidates += [base / "libmynah_asr.dylib", base / "libmynah_asr.so"]
    candidates += [Path("libmynah_asr.dylib"), Path("libmynah_asr.so")]
    for c in candidates:
        if c.exists():
            return ctypes.CDLL(str(c))
    raise OSError("libmynah_asr not found: build it with `make shared` in the repo root")


class _Word(ctypes.Structure):
    _fields_ = [("word", ctypes.c_char_p), ("t0", ctypes.c_double), ("t1", ctypes.c_double)]


_lib = None


def _api() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _find_lib()
        _lib.mynah_asr_load_quant.restype = ctypes.c_void_p
        _lib.mynah_asr_load_quant.argtypes = [ctypes.c_char_p, ctypes.c_int]
        _lib.mynah_asr_free.argtypes = [ctypes.c_void_p]
        _lib.mynah_asr_transcribe_ts.restype = ctypes.c_void_p   # char* (we own it and must free it)
        _lib.mynah_asr_transcribe_ts.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.POINTER(ctypes.POINTER(_Word)), ctypes.POINTER(ctypes.c_int)]
        _lib.mynah_asr_words_free.argtypes = [ctypes.POINTER(_Word), ctypes.c_int]
        _lib.mynah_asr_set_target_lang.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        _lib.mynah_asr_can_translate.argtypes = [ctypes.c_void_p]
        _lib.mynah_asr_can_detect_lang.argtypes = [ctypes.c_void_p]
        _lib.mynah_asr_detect_lang.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                           ctypes.c_size_t, ctypes.c_char_p]
        _lib.mynah_asr_map_lang.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        _lib.mynah_asr_set_segment_limit.argtypes = [ctypes.c_void_p, ctypes.c_double]
        _lib.mynah_asr_resample.restype = ctypes.POINTER(ctypes.c_float)
        _lib.mynah_asr_resample.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                                        ctypes.c_int, ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_size_t)]
        _lib.mynah_asr_version.restype = ctypes.c_char_p
    return _lib


def _load_wav(path: str) -> tuple[array.array, int]:
    """WAV PCM16 -> float32 [-1,1] mono (channels averaged) + sample rate."""
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("WAV PCM16 required (for mp3 and friends: ffmpeg -ar 16000 -ac 1)")
        nch, sr, n = w.getnchannels(), w.getframerate(), w.getnframes()
        pcm = array.array("h")
        pcm.frombytes(w.readframes(n))
    out = array.array("f", [0.0]) * (len(pcm) // nch)
    for i in range(len(out)):
        s = 0
        for c in range(nch):
            s += pcm[i * nch + c]
        out[i] = s / nch / 32768.0
    return out, sr


class MynahASR:
    """A loaded model. Thread-safety: use from one thread at a time."""

    def __init__(self, model_dir: str, quant: str = "f32"):
        self._lib = _api()
        self._m = self._lib.mynah_asr_load_quant(str(model_dir).encode(), QUANT[quant])
        if not self._m:
            raise RuntimeError(f"load failed: {model_dir}")

    def close(self) -> None:
        if self._m:
            self._lib.mynah_asr_free(self._m)
            self._m = None

    def __del__(self):  # noqa: D105
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def set_target_lang(self, lang: str) -> None:
        """AED/Canary: output language (different from source = translation). '' = ASR."""
        if self._lib.mynah_asr_set_target_lang(self._m, lang.encode()) != 0:
            raise ValueError(f"unsupported target lang: {lang}")

    def set_segment_limit(self, sec: float) -> None:
        self._lib.mynah_asr_set_segment_limit(self._m, sec)

    def can_detect_lang(self) -> bool:
        """True when this model REPORTS the language it heard (only Nemotron does:
        Parakeet detects it internally without saying so, and on Canary the source
        language is an input, so lang="auto" there means the default, "en")."""
        return self._lib.mynah_asr_can_detect_lang(self._m) == 1

    def detect_lang(self, wav: str) -> str | None:
        """Locale of a WAV ("it-IT"), from a short prefix — None when nothing was
        detected. Raises when the model cannot detect at all."""
        if not self.can_detect_lang():
            raise RuntimeError("this model cannot detect the language (needs an 'auto' prompt)")
        buf, n = self._samples(wav)
        return self._detect(buf, n)

    def map_lang(self, tag: str) -> str | None:
        """A detector's tag in the form THIS model takes ("it-IT" -> "it"), or None
        when it does not support that language at all."""
        out = ctypes.create_string_buffer(16)
        if self._lib.mynah_asr_map_lang(self._m, tag.encode(), out) != 0:
            return None
        return out.value.decode()

    def _samples(self, wav: str):
        samples, sr = _load_wav(wav)
        buf = (ctypes.c_float * len(samples)).from_buffer(samples)
        n = ctypes.c_size_t(len(samples))
        if sr != 16000:
            n_out = ctypes.c_size_t()
            p = self._lib.mynah_asr_resample(buf, len(samples), sr, 16000,
                                         ctypes.byref(n_out))
            if not p:
                raise RuntimeError("resampling failed")
            buf, n = p, n_out
        return buf, n

    def _detect(self, buf, n) -> str | None:
        tag = ctypes.create_string_buffer(16)
        if self._lib.mynah_asr_detect_lang(self._m, buf, n, tag) != 0:
            return None
        return tag.value.decode()

    def transcribe(self, wav: str, lang: str = "auto", lookahead: int = -1,
                   timestamps: bool = False, detector: "MynahASR | None" = None):
        """Transcribe a WAV (resampled automatically). lang also accepts "src>tgt"
        for AED translation. Returns str, or (str, [(word, t0, t1), ...], lang)
        with timestamps=True — lang is the detected language, "" when the model
        does not emit one (English-only models have no LID). Same shape as the
        Node binding's { text, words, lang }.

        detector: another MynahASR (a Nemotron) used to identify the language when
        lang is "auto" and THIS model cannot — the equivalent of the CLI's
        --lid-model. It reads a few seconds and its answer becomes the source
        language. Detecting nothing warns and leaves the model's default; detecting
        a language this model does not have raises, exactly as naming it would."""
        buf, n = self._samples(wav)
        if detector is not None and lang.split(">")[0] == "auto" and not self.can_detect_lang():
            tag = detector._detect(buf, n)
            if tag is None:
                warnings.warn(f"no language detected, falling back to this model's default"
                              f" ({wav})", stacklevel=2)
            else:
                src = self.map_lang(tag)
                if src is None:
                    raise ValueError(f"detected language {tag!r} is not supported by this model")
                tgt = lang.split(">", 1)[1] if ">" in lang else ""
                lang = f"{src}>{tgt}" if tgt else src
        lang_out = ctypes.create_string_buffer(16)
        words_p = ctypes.POINTER(_Word)()
        n_words = ctypes.c_int(0)
        raw = self._lib.mynah_asr_transcribe_ts(
            self._m, buf, n, lang.encode(), lookahead, lang_out,
            ctypes.byref(words_p) if timestamps else None,
            ctypes.byref(n_words) if timestamps else None)
        if not raw:
            raise RuntimeError("transcription failed (unsupported language?)")
        text = ctypes.cast(raw, ctypes.c_char_p).value.decode()
        libc = ctypes.CDLL(None)
        libc.free(ctypes.c_void_p(raw))
        if not timestamps:
            return text
        words = [(words_p[i].word.decode(), words_p[i].t0, words_p[i].t1)
                 for i in range(n_words.value)]
        self._lib.mynah_asr_words_free(words_p, n_words)
        return text, words, lang_out.value.decode()


def version() -> str:
    return _api().mynah_asr_version().decode()
