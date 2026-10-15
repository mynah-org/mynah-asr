"""Training-free greedy decoding sweep on a fixed checkpoint: blank-logit penalty, per-utterance dump.

Greedy RNNT (the serving decoder) with blank_logit -= P at every joint step, P over --penalties. P > 0 makes
emitting a label easier (fewer deletions, more insertions), P < 0 the reverse. P is CHOSEN on the --dev sets
(the harness) and the --report sets (clean-it) are scored for every P but must be read ONCE at the dev-chosen P.
Gate: at P = 0 this decoder must reproduce NeMo's greedy_batch WER (lm_fusion_probe.py) on every set.

Same clip choice (two_stage.py subsample, --per-set), encoder call and a-z normaliser as lm_fusion_probe.py.
Per set and P: WER, CER, S / D / I per 100 reference words, hypothesis/reference word ratio. --dump writes
one JSON line per clip (set, path, duration, ref, hyp at every P) for nbest_gap.py.

  python greedy_blank_sweep.py --nemo runs/plain-it-c250-36k/final.nemo --dev "mls=...,fleurs=...,cv=...,vp=..." \
      --report "tedx=...,italic=..." --penalties=-0.5,0,0.25,0.5,1,1.5,2 --out sweep.json --dump greedy.jsonl
"""
import argparse, json, re, time, unicodedata
import numpy as np, soundfile as sf, torch

ap = argparse.ArgumentParser()
ap.add_argument("--nemo", required=True)
ap.add_argument("--dev", required=True, help="name=manifest,... (P is chosen on their macro WER)")
ap.add_argument("--report", default="", help="name=manifest,... (scored at every P; read only at the dev-chosen P)")
ap.add_argument("--per-set", type=int, default=200, help="dev sets: two_stage.py subsample")
ap.add_argument("--report-per-set", type=int, default=100000)
ap.add_argument("--penalties", default="-0.5,0,0.25,0.5,1,1.5,2")
ap.add_argument("--max-symbols", type=int, default=10)
ap.add_argument("--pre-pad-s", type=float, default=0.0, help="digital silence prepended to every clip (cold-start test)")
ap.add_argument("--post-pad-s", type=float, default=0.0, help="digital silence appended to every clip (flush test)")
ap.add_argument("--out", default="sweep.json")
ap.add_argument("--dump", default="")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"


def norm(t):
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z' ]", " ", t)
    return " ".join(t.split())


def align(r, h):
    """-> (S, D, I) of word list h against word list r."""
    n, m_ = len(r), len(h)
    d = [[0] * (m_ + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m_ + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m_ + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    s = de = ins = 0; i, j = n, m_
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            s += r[i - 1] != h[j - 1]; i -= 1; j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            de += 1; i -= 1
        else:
            ins += 1; j -= 1
    return s, de, ins


def lev(x, y):
    prev = list(range(len(y) + 1))
    for i, p in enumerate(x, 1):
        cur = [i]
        for j, q in enumerate(y, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (p != q)))
        prev = cur
    return prev[-1]


from nemo.collections.asr.models import EncDecRNNTBPEModel

m = EncDecRNNTBPEModel.restore_from(a.nemo, map_location=dev).eval()
tk = m.tokenizer
BLANK = int(m.joint.num_classes_with_blank) - 1


def sets_of(spec, per):
    out = []
    for x in spec.split(","):
        if x:
            n, p = x.split("=", 1)
            rs = [json.loads(l) for l in open(p)]
            out.append((n, rs[:: max(1, len(rs) // per)][:per]))
    return out


dev_sets, rep_sets = sets_of(a.dev, a.per_set), sets_of(a.report, a.report_per_set)
enc = []   # (set, ref, path, duration, enc[D, T], T)
t0 = time.time()
with torch.no_grad():
    for n, rs in dev_sets + rep_sets:
        for i in range(0, len(rs), 8):
            ch = rs[i:i + 8]
            xs = []
            for r in ch:
                x, _ = sf.read(r["audio_filepath"], dtype="float32")
                x = x.mean(1) if x.ndim > 1 else x
                xs.append(np.concatenate([np.zeros(int(a.pre_pad_s * 16000), np.float32), x,
                                          np.zeros(int(a.post_pad_s * 16000), np.float32)]))
            X = torch.zeros(len(xs), max(len(x) for x in xs))
            for b, x in enumerate(xs):
                X[b, :len(x)] = torch.from_numpy(x)
            lens = torch.tensor([len(x) for x in xs])
            f, fl = m.preprocessor(input_signal=X.to(dev), length=lens.to(dev))
            e, el = m.encoder(audio_signal=f, length=fl)
            for b, r in enumerate(ch):
                enc.append((n, norm(r["text"]), r["audio_filepath"], r.get("duration", len(xs[b]) / 16000),
                            e[b, :, : int(el[b])], int(el[b])))
print(f"encoded {len(enc)} clips in {time.time() - t0:.0f}s on {dev}", flush=True)


@torch.no_grad()
def greedy(chunk, pen):
    """Batched frame-synchronous greedy RNNT, <= max_symbols labels per frame, blank logit -= pen."""
    B, T = len(chunk), max(x[5] for x in chunk)
    E = torch.zeros(B, T, chunk[0][4].shape[0], device=dev)
    for b, x in enumerate(chunk):
        E[b, : x[5]] = x[4].T
    lens = torch.tensor([x[5] for x in chunk], device=dev)
    E = m.joint.project_encoder(E)
    g, state = m.decoder.predict(None, state=None, add_sos=False, batch_size=B)[:2]
    g = m.joint.project_prednet(g)
    hyps = [[] for _ in range(B)]
    for t in range(T):
        live = lens > t
        for _ in range(a.max_symbols):
            lo = m.joint.joint_after_projection(E[:, t:t + 1], g)[:, 0, 0, :].float()
            lo[:, BLANK] -= pen
            k = lo.argmax(-1)
            emit = live & (k != BLANK)
            if not bool(emit.any()):
                break
            for b in emit.nonzero().flatten().tolist():
                hyps[b].append(int(k[b]))
            g2, s2 = m.decoder.predict(k.clamp(max=BLANK - 1)[:, None], state=state, add_sos=False, batch_size=B)[:2]
            g2 = m.joint.project_prednet(g2)
            mk = emit[:, None, None]
            g = torch.where(mk, g2, g)
            state = [torch.where(emit[None, :, None], n_, o_) for n_, o_ in zip(s2, state)]
            live = emit
    return [norm(tk.ids_to_text([y for y in h if y < 1024])) for h in hyps]


pens = [float(x) for x in a.penalties.split(",")]
order = sorted(range(len(enc)), key=lambda i: enc[i][5])   # length-sorted batches: less padding
hyp = {p: [None] * len(enc) for p in pens}
out = []
for p in pens:
    t0 = time.time()
    for i in range(0, len(order), 32):
        idx = order[i:i + 32]
        for j, h in zip(idx, greedy([enc[j] for j in idx], p)):
            hyp[p][j] = h
    rec = {"penalty": p, "sec": round(time.time() - t0, 1)}
    for n, _ in dev_sets + rep_sets:
        S = D = I = N = HW = CE = CN = 0
        for (sn, ref, *_), h in zip(enc, hyp[p]):
            if sn != n:
                continue
            rw, hw = ref.split(), h.split()
            s, d, ins = align(rw, hw)
            S += s; D += d; I += ins; N += len(rw); HW += len(hw); CE += lev(ref, h); CN += len(ref)
        rec[n] = {"wer": round(100 * (S + D + I) / N, 2), "cer": round(100 * CE / CN, 2), "S": round(100 * S / N, 2),
                  "D": round(100 * D / N, 2), "I": round(100 * I / N, 2), "len_ratio": round(HW / N, 3)}
    rec["dev_macro"] = round(float(np.mean([rec[n]["wer"] for n, _ in dev_sets])), 2)
    if rep_sets:
        rec["report_macro"] = round(float(np.mean([rec[n]["wer"] for n, _ in rep_sets])), 2)
    print(json.dumps(rec), flush=True)
    out.append(rec)
best = min(out, key=lambda r: r["dev_macro"])
summary = {"chosen_on_dev": best["penalty"], "dev_macro": best["dev_macro"], "report_macro_at_chosen": best.get("report_macro"),
           "p0": next((r for r in out if r["penalty"] == 0), None)}
print("CHOSEN", json.dumps(summary), flush=True)
json.dump({"args": vars(a), "sweep": out, "summary": summary}, open(a.out, "w"), indent=1)
if a.dump:
    with open(a.dump, "w") as fd:
        for j, (n, ref, path, dur, _, _) in enumerate(enc):
            fd.write(json.dumps({"set": n, "path": path, "duration": dur, "ref": ref,
                                 "hyp": {str(p): hyp[p][j] for p in pens}}, ensure_ascii=False) + "\n")
print("wrote", a.out)
