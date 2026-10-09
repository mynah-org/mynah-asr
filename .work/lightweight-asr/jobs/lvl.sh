#!/bin/bash
# Level-robustness DIAGNOSTIC (not a runtime change): the EOU 120M blanks on
# low-level FLEURS clips. Same 200 FLEURS EN test clips, same weights (f32),
# same scorer, after three deterministic offline transforms:
#   peak  : peak-normalise to -3 dBFS
#   rms   : RMS-normalise to -23 dBFS (clipped at 0 dBFS)
#   agc   : a simple frame AGC (300 ms RMS window toward -23 dBFS, gain capped
#           at +40 dB, 50 ms smoothing) - a causal, streamable transform
# plus Nemotron on the peak arm as a control (it should barely move).
#   tmux new -d -s lvl 'bash /root/lvl.sh 2>&1 | tee /root/res/lvl.log'
set -u
cd /root/mynah-eou || exit 2
O=/root/res/lvl; mkdir -p $O
E=models/parakeet-realtime-eou-120m; M=models/nemotron-3.5-asr-streaming-0.6b
PY=/root/mynah-asr/tools/.venv/bin/python
$PY - <<'PY'
import json, os, wave, numpy as np
src = "samples/eval-bank"; man = json.load(open(f"{src}/manifest.json"))
def rd(p):
    w = wave.open(p); return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
def wr(p, x):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    w = wave.open(p, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
def peak(x): return x * (10 ** (-3 / 20) / (np.abs(x).max() + 1e-9))
def rms(x): return x * (10 ** (-23 / 20) / (np.sqrt((x ** 2).mean()) + 1e-9))
def agc(x, sr=16000):
    win, hop = int(0.3 * sr), int(0.05 * sr); tgt = 10 ** (-23 / 20); gmax = 10 ** (40 / 20)
    g = np.ones_like(x); cur = 1.0
    for s in range(0, len(x), hop):
        seg = x[max(0, s - win):s + hop]; r = np.sqrt((seg ** 2).mean()) + 1e-9
        want = min(gmax, tgt / r); cur = 0.7 * cur + 0.3 * want; g[s:s + hop] = cur
    return x * g
for name, f in (("peak", peak), ("rms", rms), ("agc", agc)):
    root = f"/root/res/lvl/bank-{name}"
    for lang, ents in ((k, v) for k, v in man.items() if isinstance(v, list)):
        for e in ents:
            rel = e.get("file") or e.get("path") or e.get("wav")
            wr(f"{root}/{rel}", f(rd(f"{src}/{rel}")))
    json.dump(man, open(f"{root}/manifest.json", "w"))
    print("built", root)
PY
run() {  # tag model bank extra
    timeout 3600 python3 tools/eval/lang_gate.py -m $2 --quant f32 --mode stream --langs en --label $1 \
        --manifest $O/bank-$3/manifest.json --root $O/bank-$3 $4 --json $O/$1.json >$O/$1.txt 2>&1
    echo "== lvl $1 rc=$? $(date +%T)"
}
run eou-peak $E peak "" & run eou-rms $E rms "" & run eou-agc $E agc "" & run nemo-peak $M peak "" &
wait
for t in eou-peak eou-rms eou-agc nemo-peak; do echo "--- $t"; grep -E "WER |CER |pooled|empty" $O/$t.txt; done
echo "== LVL-DONE $(date +%T)"
