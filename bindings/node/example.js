'use strict';
// Node bindings example (koffi). Twin of bindings/python/example.py.
//
//   make shared          # in the repo root -> libmynah_asr.dylib/.so
//   npm i koffi
//   node bindings/node/example.js models/parakeet-tdt-0.6b-v3 audio.wav
//
// With an AED/Canary model you can translate: pass lang "src>tgt", e.g. "de>en".

const { MynahASR, version } = require('./mynah_asr');

function main() {
  const [modelDir, wav, lang = 'auto'] = process.argv.slice(2);
  if (!modelDir || !wav) {
    console.error('usage: node example.js <model_dir> <file.wav> [lang|src>tgt]');
    process.exit(2);
  }

  console.error(`libmynah_asr ${version()}`);
  const m = new MynahASR(modelDir);
  try {
    const { text, words, lang: detected } = m.transcribe(wav, { lang, timestamps: true });
    console.error(`[lang=${detected || lang}]`);   // empty on models without LID
    console.log(text);
    for (const w of words) {
      console.log(`${w.t0.toFixed(2).padStart(6)} ${w.t1.toFixed(2).padStart(6)}  ${w.word}`);
    }
  } finally {
    m.close();
  }
}

main();
