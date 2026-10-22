#!/usr/bin/env python3
"""Transcribe a 16 kHz mono WAV with the released checkpoint in NeMo, with end-of-utterance segmentation.

    pip install "nemo_toolkit[asr]>=2.6" soundfile
    python transcribe_nemo.py model.nemo audio.wav [--post-pad-s 2.0]

Decoding = the frozen characterisation protocol of the release numbers (results/): one encoder pass over the clip,
frame-synchronous greedy RNNT (max 10 symbols per 80 ms frame, no beam, no LM); after every <EOU> the decoder state
is reset (prediction network to zero, last token -> SOS) and a new segment starts. 2 s of -75 dB noise are appended
so that the final <EOU> can fire. Prints one line per segment: end time of its <EOU> (s) and the text.
The Mynah runtime additionally resets the encoder caches after the chunk holding an <EOU> (see inference.json);
transcripts differ only after a premature <EOU>.
"""
import argparse

import numpy as np
import soundfile as sf
import torch
from nemo.collections.asr.models import EncDecRNNTBPEModel

ap = argparse.ArgumentParser()
ap.add_argument("nemo")
ap.add_argument("wav")
ap.add_argument("--post-pad-s", type=float, default=2.0)
a = ap.parse_args()
m = EncDecRNNTBPEModel.restore_from(a.nemo, map_location="cpu").eval()
tk = m.tokenizer
EOU, BLANK = tk.token_to_id("<EOU>"), tk.vocab_size
x, sr = sf.read(a.wav, dtype="float32")
assert sr == 16000, "16 kHz input expected"
x = x.mean(1) if x.ndim > 1 else x
x = np.concatenate([x, (np.random.RandomState(len(x)).randn(int(a.post_pad_s * sr)) * 10 ** (-75 / 20)).astype(np.float32)])
with torch.no_grad():
    f, fl = m.preprocessor(input_signal=torch.from_numpy(x)[None], length=torch.tensor([len(x)]))
    e, el = m.encoder(audio_signal=f, length=fl)
    enc = e[0, :, : int(el[0])].T
    seg, out = [], []
    g, state = m.decoder.predict(None, state=None, add_sos=False, batch_size=1)[:2]
    for t in range(enc.shape[0]):
        for _ in range(10):
            k = int(m.joint.joint(enc[t][None, None, :], g)[0, 0, 0].argmax())
            if k == BLANK:
                break
            if k == EOU:
                out.append((t * 0.08, tk.ids_to_text(seg))); seg = []
                g, state = m.decoder.predict(None, state=None, add_sos=False, batch_size=1)[:2]
                break
            if k < 1024:   # text pieces only (<EOB> 1025 is not text)
                seg.append(k)
            g, state = m.decoder.predict(torch.tensor([[k]]), state=state, add_sos=False, batch_size=1)[:2]
if seg:
    out.append((None, tk.ids_to_text(seg)))
for t, s in out:
    print(f"[{'no <EOU>' if t is None else f'<EOU> {t:.2f} s'}] {s}")
