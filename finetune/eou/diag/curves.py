#!/usr/bin/env python3
"""Reconstruct the training dynamics of two_stage.py runs (no GPU, no torch, stdlib only).

Inputs (all read-only):
  --metrics DIR   <run>.metrics.json files (two_stage.py format: args + log[]; older
                  trainer formats with val_curve/loss_curve are carried as context)
  --logs DIR      training logs with `step N/S loss X audio H h` lines (file name = run tag
                  without the "plain-" prefix, e.g. it-b2-gradual.log)
  --stats FILE    per-source manifest stats {"<stem>@<maxdur>": {n, hours, mean_dur, ...}}
  --out FILE      curves.json

Per run and per eval it reconstructs:
  lr_mult         the LambdaLR multiplier of two_stage.py (warmup steps//20 (>=10), cosine
                  to `steps`, 1 % floor) and the absolute LR of every param group
  lr_integral     sum of the multiplier over the steps done (in full-LR step equivalents)
  enc_top_k       encoder blocks trainable (unfreeze_sched / unfreeze_top), enc_dropout
                  (encoder in train mode: only when the whole encoder is unfrozen)
  audio_h         processed hours (from the log; else bs x expected clip duration)
  src_*           expected samples, hours and passes (epochs) per source
                  (normalised weight x bs x steps; hours x mean clip duration)
  loss_blk        mean of the 25-step batch losses in (prev eval, eval]; loss_blk_per_s =
                  the same divided by the mean seconds per sample in that block (the RNNT
                  loss is reduction=mean_batch, i.e. a per-utterance sum, so it scales with
                  clip length)
  d_*             WER/CER improvement per 1000 steps, per 100 processed hours and per
                  1000 full-LR step equivalents, for the macro and each domain
Paired loss: runs that draw the same batches (same seed, same sources and weights) are
compared batch by batch (ratio of the 25-step losses, block means) -> `paired`.

    python3 curves.py --metrics metrics --logs logs --stats source_stats.json --out curves.json
"""
import argparse
import glob
import json
import math
import os
import re
from pathlib import Path

DOMAINS = ["mls", "fleurs", "cv", "vp"]
SRC2DOM = {"train_40h": "mls", "train_5h": "mls", "train_cv": "cv", "train_vp": "vp",
           "train_vp_t50": "vp", "train_fleurs": "fleurs"}
STEP_RE = re.compile(r"step (\d+)/(\d+) loss ([0-9.]+) audio ([0-9.]+) h")


def lr_mult(s, steps):
    warm = max(10, steps // 20)
    return min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))) * 0.99 + 0.01


def lr_integral(step, steps):
    return sum(lr_mult(s, steps) for s in range(step))


def unfreeze_list(a):
    if a.get("unfreeze_sched"):
        out = []
        for x in a["unfreeze_sched"].split(","):
            f, k = x.split(":")
            out.append((int(float(f) * a["steps"]), 18 if k == "all" else int(k)))
        return out
    if a.get("freeze_enc"):
        return [(0, int(a.get("unfreeze_top") or 0))]
    return [(0, 18)]


def top_k(step, sched):
    # the schedule is applied at the START of step s (before the update), i.e. after
    # `step` updates the K in force for the last update is the one at step-1
    return max([k for s, k in sched if s <= max(0, step - 1)], default=0)


def group_lrs(a, k):
    """Absolute peak LR per param group in force with top-K unfrozen (before the multiplier)."""
    base = a.get("enc_lr") or a["lr"] * 0.1
    g = {"dec_joint": a["lr"]}
    if a.get("unfreeze_sched"):
        g["enc_1_4"] = base if k >= 1 else 0.0
        g["enc_5_8"] = (a.get("enc_lr_mid") or base) if k >= 5 else 0.0
        g["enc_9_17_pre"] = (a.get("enc_lr_low") or base) if k >= 9 else 0.0
    elif a.get("freeze_enc"):
        g["enc_top"] = base if k else 0.0
    else:
        g["enc_all"] = base
    return g


def parse_log(path):
    pts = []
    for line in open(path, errors="replace"):
        m = STEP_RE.search(line)
        if m:
            pts.append((int(m.group(1)), float(m.group(3)), float(m.group(4))))
    return pts


def load_sources(a, stats):
    md = int(a.get("max_dur") or 20)
    if a.get("mix"):
        items = [x.rsplit(":", 1) for x in a["mix"].split(",")]
    else:
        items = [(a.get("manifest") or "", "1.0")]
    out, tot = [], sum(float(w) for _, w in items)
    for p, w in items:
        stem = Path(p).stem
        key = {"yg_agree<=15": "yg_agree15", "yg_rand_eq<=15": "yg_rand15"}.get(stem, stem)
        st = stats.get(f"{key}@{md}") or stats.get(f"{key}@20") or {}
        out.append({"name": stem, "weight": float(w) / tot, "n": st.get("n"), "hours": st.get("hours"),
                    "mean_dur": st.get("mean_dur"), "mean_words": st.get("mean_words"),
                    "domain": SRC2DOM.get(stem)})
    return out


def rnd(x, n=3):
    return None if x is None else round(x, n)


def build_run(name, d, stats, logs_dir):
    a = d["args"]
    steps, bs = a["steps"], a["bs"]
    sched = unfreeze_list(a)
    srcs = load_sources(a, stats)
    exp_s_per_sample = sum(s["weight"] * (s["mean_dur"] or 0) for s in srcs) or None
    tag = name[len("plain-"):] if name.startswith("plain-") else name
    lp = Path(logs_dir) / f"{tag}.log"
    pts = parse_log(lp) if lp.exists() else []
    run = {"run": name, "kind": "two_stage", "steps": steps, "bs": bs, "lr": a["lr"], "enc_lr": a.get("enc_lr"),
           "enc_lr_mid": a.get("enc_lr_mid"), "enc_lr_low": a.get("enc_lr_low"),
           "unfreeze": sched, "init": a.get("init") or "stock", "eou_append": a.get("eou_append"),
           "max_dur": a.get("max_dur"), "val_n": a.get("val_n"), "sources": srcs,
           "expected_s_per_sample": rnd(exp_s_per_sample), "log_file": lp.name if pts else None,
           "audio_h_seen_final": d.get("audio_h_seen"), "wall_s": d.get("wall_s"),
           "loss_curve": [], "evals": []}
    # 25-step loss curve with a centred 11-point (275-step) moving average and per-second loss
    for i, (s, l, h) in enumerate(pts):
        win = pts[max(0, i - 5): i + 6]
        sm = sum(x[1] for x in win) / len(win)
        prev = pts[i - 1] if i else (0, 0, 0.0)
        sps = (h - prev[2]) * 3600 / ((s - prev[0]) * bs) if s > prev[0] else None
        run["loss_curve"].append({"step": s, "loss": l, "audio_h": h, "loss_ma11": rnd(sm),
                                  "s_per_sample": rnd(sps, 2), "lr_mult": rnd(lr_mult(s - 1, steps), 4),
                                  "enc_top_k": top_k(s, sched)})
    seen = {}
    log = [r for r in d["log"]]
    if a.get("eval_only"):
        log = log[:1]
    prev = None
    for r in log:
        s = r["step"]
        k = top_k(s, sched)
        aud = None
        for ps, _, ph in pts:
            if ps == s:
                aud = ph
        if aud is None:
            aud = s * bs * (exp_s_per_sample or 0) / 3600 if exp_s_per_sample else None
            if s == steps and d.get("audio_h_seen"):
                aud = d["audio_h_seen"]
        blk = [x for x in pts if (prev["step"] if prev else 0) < x[0] <= s]
        loss_blk = sum(x[1] for x in blk) / len(blk) if blk else None
        if blk:
            p0 = [x for x in pts if x[0] <= (prev["step"] if prev else 0)]
            h0 = p0[-1][2] if p0 else 0.0
            s0 = p0[-1][0] if p0 else 0
            sps_blk = (blk[-1][2] - h0) * 3600 / ((blk[-1][0] - s0) * bs)
        else:
            sps_blk = None
        lm = lr_mult(max(0, s - 1), steps)
        e = {"step": s, "lr_mult": rnd(lm, 4), "lr_integral": rnd(lr_integral(s, steps), 1),
             "lr_abs": {g: float(f"{v * lm:.3g}") for g, v in group_lrs(a, k).items()},
             "enc_top_k": k, "enc_dropout": k > 17, "audio_h": rnd(aud),
             "loss_at_eval": r.get("loss"), "loss_blk": rnd(loss_blk), "s_per_sample_blk": rnd(sps_blk, 2),
             "loss_blk_per_s": rnd(loss_blk / sps_blk if loss_blk and sps_blk else None),
             "train8_wer": r.get("train8_wer"), "train8_cer": r.get("train8_cer"),
             "nonblank_frac": r.get("nonblank_frac"), "margin_max_mean": r.get("margin_max_mean"),
             "val_wer": r.get("val_wer"), "val_cer": r.get("val_cer"), "val_empty": r.get("val_empty"),
             "val_eou_rate": r.get("val_eou_rate"), "domains": {}, "sources": {}}
        for dn in DOMAINS:
            if f"val_{dn}_wer" in r:
                e["domains"][dn] = {k2: r.get(f"val_{dn}_{k2}") for k2 in ("wer", "cer", "empty", "eou")}
        if not e["domains"] and r.get("val_wer") is not None:   # single-val runs: the val set is named by args.val
            v = (a.get("val") or "")
            dn = "mls" if "mls" in v else "fleurs" if "fleurs" in v else "val"
            e["domains"][dn] = {"wer": r.get("val_wer"), "cer": r.get("val_cer"), "empty": r.get("val_empty"),
                                "eou": r.get("val_eou_rate")}
        dom_h = {}
        for sc in srcs:
            n_s = sc["weight"] * bs * s
            hh = n_s * (sc["mean_dur"] or 0) / 3600
            e["sources"][sc["name"]] = {"samples": round(n_s), "hours": rnd(hh, 2),
                                        "passes": rnd(n_s / sc["n"], 3) if sc["n"] else None}
            if sc["domain"]:
                dom_h[sc["domain"]] = dom_h.get(sc["domain"], 0) + hh
        e["domain_hours_seen"] = {k2: rnd(v, 2) for k2, v in dom_h.items()}
        if prev is not None and s > prev["step"]:
            ds = s - prev["step"]
            dh = (e["audio_h"] or 0) - (prev["audio_h"] or 0)
            dl = e["lr_integral"] - prev["lr_integral"]
            dd = {}
            for key, cur, old in [("macro_wer", e["val_wer"], prev["val_wer"]), ("macro_cer", e["val_cer"], prev["val_cer"])] + \
                    [(f"{dn}_wer", e["domains"][dn]["wer"], prev["domains"].get(dn, {}).get("wer")) for dn in e["domains"]]:
                if cur is None or old is None:
                    continue
                imp = old - cur
                dd[key] = {"per_1k_steps": rnd(imp * 1000 / ds, 2), "per_100h": rnd(imp * 100 / dh, 2) if dh > 0 else None,
                           "per_1k_lr_steps": rnd(imp * 1000 / dl, 2) if dl > 0 else None}
            e["d"] = dd
        run["evals"].append(e)
        prev = e
    return run


def paired(runs, a_name, b_name):
    """Batch-paired loss ratio b/a per 500-step block (same seed and batch stream)."""
    A = {p["step"]: p["loss"] for p in runs[a_name]["loss_curve"]}
    B = {p["step"]: p["loss"] for p in runs[b_name]["loss_curve"]}
    common = sorted(set(A) & set(B))
    out = []
    for lo in range(0, max(common, default=0), 250):
        ss = [s for s in common if lo < s <= lo + 250]
        if not ss:
            continue
        la, lb = sum(A[s] for s in ss), sum(B[s] for s in ss)
        out.append({"block": f"{lo + 1}-{lo + 250}", "ratio_b_over_a": round(lb / la, 4),
                    "median_ratio": round(sorted(B[s] / A[s] for s in ss)[len(ss) // 2], 4), "n": len(ss)})
    return {"a": a_name, "b": b_name, "blocks": out}


def context_run(name, d):
    """Older trainer formats (Canary AED arms, the collapsed NeMo-trainer EOU runs): eval curve only."""
    h = d.get("hparams", {})
    vc = []
    for v in d.get("val_curve", []):
        x = {"step": v["step"], "audio_h": v.get("audio_h")}
        for k, val in v.items():
            if k.startswith("eval_") or k == "_pooled":
                x[k] = val[0] if isinstance(val, list) else val
        vc.append(x)
    return {"run": name, "kind": "context_other_trainer", "init": d.get("init") or h.get("stock"),
            "hparams": {k: h.get(k) for k in ("subset", "epochs", "lr", "enc_lr_scale", "top_n", "freeze_encoder_layers",
                                              "warmup_frac", "batch_duration", "val_n")},
            "max_steps": d.get("max_steps"), "audio_hours_processed": d.get("audio_hours_processed"),
            "equiv_epochs": d.get("equiv_epochs") or d.get("equiv_epochs_padded"),
            "val_curve": vc, "loss_curve": d.get("loss_curve")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default="plain-it-a2-vpt50:plain-it-b2-gradual,plain-it-a2-vpt50:plain-it-b1-top8,"
                                       "plain-it-c2-ygrand:plain-it-c1-yg15,plain-it-a0-mix:plain-it-a1-gain10")
    a = ap.parse_args()
    stats = json.load(open(a.stats))
    runs = {}
    for f in sorted(glob.glob(os.path.join(a.metrics, "*.metrics.json"))):
        name = os.path.basename(f)[: -len(".metrics.json")]
        d = json.load(open(f))
        runs[name] = build_run(name, d, stats, a.logs) if "args" in d else context_run(name, d)
    pairs = []
    for p in a.pairs.split(","):
        x, y = p.split(":")
        if x in runs and y in runs and runs[x].get("loss_curve") and runs[y].get("loss_curve"):
            pairs.append(paired(runs, x, y))
    meta = {"lr_schedule": "mult(s) = min(1,(s+1)/max(10,steps//20)) * 0.5*(1+cos(pi*min(1,s/steps))) * 0.99 + 0.01; "
                           "same multiplier on every param group",
            "loss_note": "25-step losses are single-batch values (reduction mean_batch = per-utterance sum); "
                         "compare runs only batch-paired or per second of audio",
            "source_stats": stats}
    json.dump({"meta": meta, "runs": runs, "paired_loss": pairs}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(runs)} runs, {len(pairs)} paired comparisons")


if __name__ == "__main__":
    main()
