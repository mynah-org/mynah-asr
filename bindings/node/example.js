'use strict';
// Node bindings example (koffi). Twin of bindings/python/example.py.
//
//   make shared          # in the repo root -> libmynah.dylib/.so
//   npm i koffi
//   node bindings/node/example.js models/parakeet-tdt-0.6b-v3 audio.wav
//
// With an AED/Canary model you can translate: pass lang "src>tgt", e.g. "de>en".

const { Mynah, version } = require('./mynah');

function main() {
  const [modelDir, wav, lang = 'auto'] = process.argv.slice(2);
  if (!modelDir || !wav) {
    console.error('usage: node example.js <model_dir> <file.wav> [lang|src>tgt]');
    process.exit(2);
  }

  console.error(`libmynah ${version()}`);
  const m = new Mynah(modelDir);
  try {
    const { text, words, lang: detected } = m.transcribe(wav, { lang, timestamps: true });
    console.error(`[lang=${detected}]`);
    console.log(text);
    for (const w of words) {
      console.log(`${w.t0.toFixed(2).padStart(6)} ${w.t1.toFixed(2).padStart(6)}  ${w.word}`);
    }
  } finally {
    m.close();
  }
}

main();
