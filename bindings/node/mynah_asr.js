'use strict';
// Node bindings for libmynah_asr (koffi, pure FFI — NO build step in the repo).
//
// Prerequisites:
//   1) `make shared` in the repo root      -> libmynah_asr.dylib / libmynah_asr.so
//   2) `npm i koffi`                        (wherever you like: no native build)
//
//   const { MynahASR } = require('./mynah_asr');
//   const m = new MynahASR('models/parakeet-tdt-0.6b-v3');
//   console.log(m.transcribe('audio.wav'));
//   const { text, words, lang } = m.transcribe('audio.wav', { timestamps: true });
//   // translation (AED/Canary models): lang "src>tgt"
//   console.log(new MynahASR('models/canary-180m-flash').transcribe('de.wav', { lang: 'de>en' }));
//   m.close();
//
// Twin of bindings/python/mynah.py: same surface, same 7 symbols.

const fs = require('fs');
const path = require('path');

let koffi;
try {
  koffi = require('koffi');
} catch {
  throw new Error("koffi not installed: `npm i koffi` (FFI only, no native build)");
}

const QUANT = { f32: 0, int8: 1, int4: 2 };

function findLib() {
  const names = process.platform === 'darwin' ? ['libmynah_asr.dylib']
              : process.platform === 'win32' ? ['mynah_asr.dll', 'libmynah_asr.dll']
              : ['libmynah_asr.so'];
  const bases = [__dirname, path.resolve(__dirname, '..', '..'), process.cwd()];
  for (const b of bases) {
    for (const n of names) {
      const p = path.join(b, n);
      if (fs.existsSync(p)) return p;
    }
  }
  throw new Error('libmynah_asr not found: build it with `make shared` in the repo root');
}

const lib = koffi.load(findLib());

// libc.free: the text returned by mynah_asr_transcribe_ts is malloc'd by the runtime and
// must be released with the same free (like ctypes.CDLL(None) in the Python binding).
const libc = koffi.load(
  process.platform === 'darwin' ? 'libSystem.B.dylib'
  : process.platform === 'win32' ? 'msvcrt.dll'
  : 'libc.so.6');
const cfree = libc.func('void free(void *ptr)');

const Word = koffi.struct('mynah_asr_word', { word: 'char *', t0: 'double', t1: 'double' });

const fn = {
  load_quant: lib.func('void *mynah_asr_load_quant(const char *dir, int quant)'),
  free: lib.func('void mynah_asr_free(void *m)'),
  version: lib.func('const char *mynah_asr_version()'),
  set_target_lang: lib.func('int mynah_asr_set_target_lang(void *m, const char *lang)'),
  can_translate: lib.func('int mynah_asr_can_translate(void *m)'),
  set_segment_limit: lib.func('void mynah_asr_set_segment_limit(void *m, double sec)'),
  words_free: lib.func('void mynah_asr_words_free(mynah_asr_word *w, int n)'),
  resample: lib.func(
    'float *mynah_asr_resample(float *in, size_t n, int sr_in, int sr_out, _Out_ size_t *n_out)'),
  // same symbol, two prototypes: text-only (words = NULL) and with timestamps.
  // lang_out is an output buffer: uint8_t* (a real pointer) and not char*, which
  // koffi would treat as an input string, dropping what C writes back.
  transcribe_text: lib.func(
    'void *mynah_asr_transcribe_ts(void *m, float *s, size_t n, const char *lang, int la, ' +
    'uint8_t *lang_out, void *words, void *n_words)'),
  transcribe_ts: lib.func(
    'void *mynah_asr_transcribe_ts(void *m, float *s, size_t n, const char *lang, int la, ' +
    'uint8_t *lang_out, _Out_ mynah_asr_word **words, _Out_ int *n_words)'),
};

// ---- WAV PCM16 -> Float32Array mono [-1,1] + sample rate (minimal parser) ----
function loadWav(file) {
  const b = fs.readFileSync(file);
  if (b.toString('latin1', 0, 4) !== 'RIFF' || b.toString('latin1', 8, 12) !== 'WAVE') {
    throw new Error('not a RIFF/WAVE file');
  }
  let off = 12, fmt = null, data = null;
  while (off + 8 <= b.length) {
    const id = b.toString('latin1', off, off + 4);
    const size = b.readUInt32LE(off + 4);
    const body = off + 8;
    if (id === 'fmt ') {
      fmt = {
        audioFormat: b.readUInt16LE(body),
        channels: b.readUInt16LE(body + 2),
        sampleRate: b.readUInt32LE(body + 4),
        bits: b.readUInt16LE(body + 14),
      };
    } else if (id === 'data') {
      data = b.subarray(body, body + size);
    }
    off = body + size + (size & 1); // chunks are word-aligned
  }
  if (!fmt || !data) throw new Error('WAV without fmt/data chunks');
  if (fmt.bits !== 16) throw new Error('WAV PCM16 required (for mp3 and friends: ffmpeg -ar 16000 -ac 1)');
  const nch = fmt.channels;
  const nSamp = Math.floor(data.length / 2 / nch);
  const out = new Float32Array(nSamp);
  for (let i = 0; i < nSamp; i++) {
    let s = 0;
    for (let c = 0; c < nch; c++) s += data.readInt16LE((i * nch + c) * 2);
    out[i] = s / nch / 32768.0;
  }
  return { samples: out, sampleRate: fmt.sampleRate };
}

class MynahASR {
  // A loaded model. Thread-safety: use from one thread at a time.
  constructor(modelDir, quant = 'f32') {
    if (!(quant in QUANT)) throw new Error(`unknown quant: ${quant}`);
    this._m = fn.load_quant(String(modelDir), QUANT[quant]);
    if (!this._m) throw new Error(`load failed: ${modelDir}`);
  }

  close() {
    if (this._m) { fn.free(this._m); this._m = null; }
  }

  setTargetLang(lang) {
    // AED/Canary: output language (different from source = translation). '' = ASR.
    if (fn.set_target_lang(this._m, lang) !== 0) throw new Error(`unsupported target lang: ${lang}`);
  }

  canTranslate() { return fn.can_translate(this._m) === 1; }

  setSegmentLimit(sec) { fn.set_segment_limit(this._m, sec); }

  // Transcribe a WAV (resampled to 16 kHz automatically). opts: { lang, lookahead, timestamps }.
  // lang accepts "src>tgt" for AED translation. Returns a string, or
  // { text, words: [{ word, t0, t1 }, ...] } with timestamps: true.
  transcribe(wav, opts = {}) {
    const { lang = 'auto', lookahead = -1, timestamps = false } = opts;
    let { samples, sampleRate } = loadWav(wav);
    let n = samples.length;

    if (sampleRate !== 16000) {
      const nOut = [0n];
      const p = fn.resample(samples, n, sampleRate, 16000, nOut);
      if (!p) throw new Error('resampling failed');
      const count = Number(nOut[0]);
      samples = Float32Array.from(koffi.decode(p, 'float', count));
      cfree(p);
      n = count;
    }

    const langOut = Buffer.alloc(16);
    let raw;
    let words = null;
    if (timestamps) {
      const wp = [null];
      const nw = [0];
      raw = fn.transcribe_ts(this._m, samples, n, lang, lookahead, langOut, wp, nw);
      if (!raw) throw new Error('transcription failed (unsupported language?)');
      if (wp[0] && nw[0] > 0) {
        const arr = koffi.decode(wp[0], 'mynah_asr_word', nw[0]);
        words = arr.map((w) => ({ word: w.word, t0: w.t0, t1: w.t1 }));
        fn.words_free(wp[0], nw[0]);
      } else {
        words = [];
      }
    } else {
      raw = fn.transcribe_text(this._m, samples, n, lang, lookahead, langOut, null, null);
      if (!raw) throw new Error('transcription failed (unsupported language?)');
    }

    const text = koffi.decode(raw, 'char', -1); // NUL-terminated C string from the pointer
    cfree(raw);                                 // released with the runtime's free (malloc)
    return timestamps ? { text, words, lang: cstr(langOut) } : text;
  }
}

function cstr(buf) {
  const z = buf.indexOf(0);
  return buf.toString('utf8', 0, z < 0 ? buf.length : z);
}

function version() {
  return fn.version();
}

module.exports = { MynahASR, version, loadWav };
