#!/usr/bin/env python3
"""Unit tests of gpu/tools/ab/ab_report.py on synthetic sessions (CPU, stdlib only).

    python3 gpu/tools/ab/test_ab_report.py [-v]

Every session is written the way ab_server.sh / ab_device.sh write one (ab.json,
meta-<run>.json, load-<run>.json or bench-<run>.txt), with the latencies planted,
so each gate and each verdict rule is exercised by a case built to trip it."""
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_report  # noqa: E402

def _dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)


CLIPS = [f"samples/xx/clip{i}.wav" for i in range(5)]


def utt(clip, stream, rep, lag, fin, ttfp, onset=None, text=None, outcome="ok", t=0.0):
    u = {"clip": clip, "stream": stream, "rep": rep, "outcome": outcome,
         "error": None if outcome == "ok" else outcome,
         "error_kind": None if outcome == "ok" else outcome,
         "text": text if text is not None else f"text of {clip}", "audio_s": 10.0,
         "t_start": t, "t_end": t + 11.0,
         "lag_marks": [[t + k, lag + (k % 3)] for k in range(10)],
         "fin_ms": fin, "ttfp_ms": ttfp,
         "ttfp_from_onset_ms": None if onset is None else ttfp - onset}
    return u


def load_json(streams, repeat, lag, fin, ttfp, onset=200.0, steps=1000, override=None,
              errors_total=0):
    utts = []
    for s in range(streams):
        for k in range(repeat):
            clip = CLIPS[(s + k * streams) % len(CLIPS)]
            u = utt(clip, s, k, lag, fin, ttfp, onset)
            if override:
                u = override(u) or u
            utts.append(u)
    hb = {"status": "ok", "steps": 0, "balanced": True,
          "batch": {"batched_steps_total": 0, "ready_sum": 0,
                    "step_wall_ms": {"sum": 0.0, "count": 0}},
          "engine": {"device_errors_total": 0}}
    ha = {"status": "ok", "steps": steps, "balanced": True,
          "batch": {"batched_steps_total": steps, "ready_sum": steps * 4,
                    "step_wall_ms": {"sum": steps * 3.0, "count": steps}},
          "engine": {"device_errors_total": 0}}
    return {"manifest": {"health_before": hb, "health_after": ha},
            "summary": {"counts": {"errors_total": errors_total, "stream_deaths": 0,
                                   "rejected_total": 0},
                        "accounting": {"conservation": True},
                        "metrics": {"audio_per_wall": {"max": streams * 0.9}}},
            "envelope": {"verdict": "PASS"}, "utterances": utts}


class Session:
    """A synthetic session directory. arm_values: {arm: [per-occurrence dict of knobs]}."""

    def __init__(self, root, tag, order="ABBA", levels=(16,), kind="server", repeat=2):
        self.d = os.path.join(root, tag)
        os.makedirs(self.d)
        self.kind, self.order, self.levels, self.repeat = kind, order, levels, repeat
        arms = {}
        for c in order:
            arms[c] = {"label": f"arm{c}", "bin": "/x", "bin_sha": "0", "args": ""}
        first = []
        for c in order:
            if c not in first:
                first.append(c)
        self.plan, seq = [], 0
        for lvl in levels:
            for pos, arm in enumerate(order):
                seq += 1
                self.plan.append({"run": f"C{lvl:03d}-{seq:02d}-{arm}", "seq": seq, "level": lvl,
                                  "arm": arm, "cycle": 0, "pos": pos,
                                  "streams": lvl if kind == "server" else None,
                                  "repeat": repeat if kind == "server" else None})
        _dump({"kind": kind, "tag": tag, "order": order, "arms_order": first, "arms": arms,
               "levels": list(levels), "plan": self.plan, "extra": {"require_cuda": "1"}},
              os.path.join(self.d, "ab.json"))

    def write(self, fn, meta=None, skip=()):
        """fn(run) -> load JSON (server) or bench text (device); None = no file."""
        for r in self.plan:
            if r["run"] in skip:
                continue
            m = {"run": r["run"], "level": r["level"], "arm": r["arm"]}
            if self.kind == "server":
                m.update(ready=True, timed_out=False, server_alive_after=True,
                         health_status="ok", engine_cuda=True, load_rc=0)
            else:
                m.update(rc=0, timed_out=False)
            if meta:
                m.update(meta(r) or {})
            _dump(m, os.path.join(self.d, f"meta-{r['run']}.json"))
            body = fn(r)
            if body is None:
                continue
            if self.kind == "server":
                _dump(body, os.path.join(self.d, f"load-{r['run']}.json"))
            else:
                with open(os.path.join(self.d, f"bench-{r['run']}.txt"), "w") as f:
                    f.write(body)
        return self.d


def bench_text(C, ms, att=1.0):
    lines = [f"BENCH ready C={C} precision=f32 gemm=own vram_weights_mib=1 vram_arena_mib=1 vram_used_mib=2",
             f"BENCH C={C} full_steps=38 passes=38 device_ms_per_pass={ms:.3f} step_wall_ms={ms + 0.5:.3f}"]
    for s in ab_report.STAGES:
        lines.append(f"BENCH   {s:<10} {att if s == 'att_core' else 0.2:8.3f} ms/pass")
    return "\n".join(lines) + "\n"


def run(d, *extra):
    argv = [d, "--json", os.path.join(d, "v.json")] + list(extra)
    out = io.StringIO()
    old = sys.stdout
    sys.stdout = out
    try:
        rc = ab_report.main(argv)
    finally:
        sys.stdout = old
    with open(os.path.join(d, "v.json")) as f:
        return rc, json.load(f), out.getvalue()


def nth(r):
    """0 for the first occurrence of the run's arm in the order, 1 for the second."""
    return 0 if r["seq"] <= 2 else 1


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    # ---------------------------------------------------------------- gate 3 verdicts
    def test_better_arm_passes(self):
        s = Session(self.root, "better")
        lag = {"A": [100, 104], "B": [70, 72]}
        d = s.write(lambda r: load_json(16, 2, lag[r["arm"]][nth(r)], 300, 900))
        rc, v, txt = run(d)
        self.assertEqual(rc, 0)
        self.assertTrue(v["gates_pass"])
        c = v["levels_report"]["16"]["compare"]["B"]["lag_p95"]
        self.assertEqual(c["verdict"], "BETTER")
        self.assertEqual(v["verdicts"]["B"], "PASS")
        self.assertEqual(v["winner"]["arm"], "B")
        self.assertIn("VERDICT", txt)

    def test_effect_inside_noise_is_null(self):
        s = Session(self.root, "noise")
        # AA spread 20 ms, effect 10 ms: below 2 x floor
        lag = {"A": [100, 120], "B": [105, 115]}
        d = s.write(lambda r: load_json(16, 2, lag[r["arm"]][nth(r)], 300, 900))
        rc, v, _ = run(d)
        self.assertEqual(rc, 0)
        self.assertEqual(v["levels_report"]["16"]["compare"]["B"]["lag_p95"]["verdict"], "NULL")
        self.assertEqual(v["verdicts"]["B"], "NULL")
        self.assertEqual(v["winner"]["arm"], "A")

    def test_effect_not_replicated_in_both_pairs_is_null(self):
        s = Session(self.root, "onepair")
        # floor 5, threshold 10; the median effect is 11 (past it), but pair 1 gains 16
        # and pair 2 only 6: the effect is carried by one pair, so it is not replicated
        lag = {"A": [105, 100], "B": [89, 94]}
        d = s.write(lambda r: load_json(16, 2, lag[r["arm"]][nth(r)], 300, 900))
        rc, v, _ = run(d)
        c = v["levels_report"]["16"]["compare"]["B"]["lag_p95"]
        self.assertAlmostEqual(c["threshold"], 10.0)
        self.assertAlmostEqual(c["effect"], -11.0)
        self.assertEqual(c["pairs"], [-16, -6])
        self.assertEqual(c["verdict"], "NULL", c)

    def test_abs_floor_blocks_a_zero_spread_win(self):
        s = Session(self.root, "zerospread")
        lag = {"A": [100, 100], "B": [97, 97]}       # 3 ms < 5 ms absolute minimum
        d = s.write(lambda r: load_json(16, 2, lag[r["arm"]][nth(r)], 300, 900))
        _, v, _ = run(d)
        self.assertEqual(v["levels_report"]["16"]["compare"]["B"]["lag_p95"]["verdict"], "NULL")

    def test_worse_primary_fails(self):
        s = Session(self.root, "worse")
        fin = {"A": [300, 305], "B": [400, 410]}
        d = s.write(lambda r: load_json(16, 2, 100, fin[r["arm"]][nth(r)], 900))
        rc, v, _ = run(d)
        self.assertEqual(rc, 0)                      # the gates held: a real result
        self.assertEqual(v["levels_report"]["16"]["compare"]["B"]["fin_p95"]["verdict"], "WORSE")
        self.assertEqual(v["verdicts"]["B"], "FAIL")

    def test_better_and_worse_is_fail(self):
        s = Session(self.root, "mixed")
        lag = {"A": [100, 100], "B": [60, 60]}
        fin = {"A": [300, 300], "B": [400, 400]}
        d = s.write(lambda r: load_json(16, 2, lag[r["arm"]][nth(r)], fin[r["arm"]][nth(r)], 900))
        _, v, _ = run(d)
        self.assertEqual(v["verdicts"]["B"], "FAIL")

    def test_onset_ttfp_is_primary_when_given_raw_otherwise(self):
        s = Session(self.root, "onset")
        d = s.write(lambda r: load_json(16, 2, 100, 300, 900))
        _, v, _ = run(d)
        self.assertIn("ttfp_on_p95", v["primary"])
        self.assertNotIn("ttfp_p95", v["primary"])
        s2 = Session(self.root, "noonset")
        d2 = s2.write(lambda r: load_json(16, 2, 100, 300, 900, onset=None))
        _, v2, _ = run(d2)
        self.assertIn("ttfp_p95", v2["primary"])
        med = v["levels_report"]["16"]["medians"]["A"]
        self.assertAlmostEqual(med["ttfp_on_p95"], 700.0)
        self.assertAlmostEqual(med["lanes_mean"], 4.0)
        self.assertAlmostEqual(med["step_wall_ms"], 3.0)
        self.assertEqual(med["steps"], 1000)

    def test_percentiles_are_from_the_utterances(self):
        s = Session(self.root, "pct")

        def body(r):
            d = load_json(16, 2, 100, 300, 900)
            for i, u in enumerate(d["utterances"]):
                u["fin_ms"] = float(i + 1)         # 1..32
            return d
        d = s.write(body)
        _, v, _ = run(d)
        med = v["levels_report"]["16"]["medians"]["A"]
        self.assertEqual(med["fin_p50"], 16.0)     # nearest rank of 32 samples
        self.assertEqual(med["fin_p95"], 31.0)
        self.assertEqual(med["fin_p99"], 32.0)

    def test_three_arm_sweep_and_winner(self):
        s = Session(self.root, "sweep", order="ABCCBA", levels=(16, 64))
        # A = cohort 40 (baseline), B = 0 (worse finalisation), C = 20 (better lag)
        lag = {"A": 100, "B": 100, "C": 80}
        fin = {"A": 300, "B": 380, "C": 300}

        def body(r):
            j = 0 if r["pos"] < 3 else 1
            return load_json(r["level"], 2, lag[r["arm"]] + 2 * j, fin[r["arm"]] + 2 * j, 900)
        d = s.write(body)
        rc, v, txt = run(d)
        self.assertEqual(rc, 0)
        self.assertEqual(v["verdicts"], {"B": "FAIL", "C": "PASS"})
        self.assertEqual(v["winner"]["arm"], "C")
        # mirror pairs: A's runs are positions 0 and 5
        self.assertEqual(v["levels_report"]["16"]["runs"][0], "C016-01-A")
        self.assertIn("WINNER", txt)

    # ---------------------------------------------------------------- gate 1
    def test_identity_across_arms_fails_and_hides_latency(self):
        s = Session(self.root, "ident")

        def body(r):
            def ov(u):
                if r["arm"] == "B" and u["clip"] == CLIPS[2]:
                    u["text"] = "a different transcript"
            return load_json(16, 2, 100 if r["arm"] == "A" else 50, 300, 900, override=ov)
        d = s.write(body)
        rc, v, txt = run(d)
        self.assertEqual(rc, 2)
        self.assertFalse(v["gates"]["identity"]["pass"])
        self.assertEqual(v["levels_report"], {})
        self.assertIn("NOT EVALUATED", txt)
        self.assertIn(CLIPS[2], txt)

    def test_identity_arm_mode_tolerates_a_cross_arm_change(self):
        s = Session(self.root, "identarm")

        def body(r):
            def ov(u):
                if r["arm"] == "B" and u["clip"] == CLIPS[2]:
                    u["text"] = "a different transcript"
            return load_json(16, 2, 100, 300, 900, override=ov)
        d = s.write(body)
        rc, v, _ = run(d, "--identity", "arm")
        self.assertEqual(rc, 0)
        self.assertTrue(v["gates"]["identity"]["pass"])
        self.assertEqual(len(v["gates"]["identity"]["warnings"]), 1)

    def test_identity_within_a_run_across_streams_fails(self):
        s = Session(self.root, "identrun")

        def body(r):
            def ov(u):
                # stream 7 of one A run disagrees with stream 2 on the same clip
                if r["run"] == "C016-01-A" and u["stream"] == 7 and u["rep"] == 0:
                    u["text"] = "flipped"
            return load_json(16, 2, 100, 300, 900, override=ov)
        d = s.write(body)
        rc, v, _ = run(d, "--identity", "arm")       # even the relaxed mode
        self.assertEqual(rc, 2)
        fails = v["gates"]["identity"]["failures"]
        self.assertTrue(any("within run C016-01-A" in f for f in fails), fails)

    # ---------------------------------------------------------------- gate 2
    def test_error_utterance_fails_gate2(self):
        s = Session(self.root, "err")

        def body(r):
            def ov(u):
                if r["run"] == "C016-03-B" and u["stream"] == 0 and u["rep"] == 1:
                    u["outcome"] = "timeout"
                    u["error"] = "timeout waiting for done"
            return load_json(16, 2, 100, 300, 900, override=ov)
        d = s.write(body)
        rc, v, txt = run(d)
        self.assertEqual(rc, 2)
        self.assertTrue(v["gates"]["identity"]["pass"])
        self.assertFalse(v["gates"]["integrity"]["pass"])
        self.assertIn("timeout", txt)
        self.assertEqual(v["levels_report"], {})

    def test_server_death_and_missing_runs_fail_gate2(self):
        s = Session(self.root, "died")
        d = s.write(lambda r: None if r["run"] == "C016-02-B" else load_json(16, 2, 100, 300, 900),
                    meta=lambda r: {"server_alive_after": False} if r["run"] == "C016-02-B" else None,
                    skip=("C016-04-A",))
        rc, v, _ = run(d)
        self.assertEqual(rc, 2)
        probs = " | ".join(v["gates"]["integrity"]["problems"])
        self.assertIn("C016-02-B: server died", probs)
        self.assertIn("C016-02-B: no load JSON", probs)
        self.assertIn("C016-04-A: never ran", probs)

    def test_lost_utterances_and_counts_fail_gate2(self):
        s = Session(self.root, "lost")

        def body(r):
            d = load_json(16, 2, 100, 300, 900, errors_total=1 if r["arm"] == "A" else 0)
            if r["run"] == "C016-03-B":
                d["utterances"] = d["utterances"][:-1]   # one record never came back
            return d
        d = s.write(body)
        rc, v, _ = run(d)
        probs = " | ".join(v["gates"]["integrity"]["problems"])
        self.assertEqual(rc, 2)
        self.assertIn("counts.errors_total = 1", probs)
        self.assertIn("31 utterance records, expected 32", probs)

    def test_not_cuda_banner_fails_gate2(self):
        s = Session(self.root, "nocuda")
        d = s.write(lambda r: load_json(16, 2, 100, 300, 900),
                    meta=lambda r: {"engine_cuda": False} if r["arm"] == "B" else None)
        rc, v, _ = run(d)
        self.assertEqual(rc, 2)
        self.assertIn("engine=cuda", " ".join(v["gates"]["integrity"]["problems"]))

    def test_force_latency_is_labelled_diagnostic(self):
        s = Session(self.root, "force")
        d = s.write(lambda r: load_json(16, 2, 100, 300, 900, errors_total=1))
        rc, v, txt = run(d, "--force-latency")
        self.assertEqual(rc, 2)
        self.assertIn("DIAGNOSTIC ONLY", txt)
        self.assertNotIn("== VERDICT", txt)

    # ---------------------------------------------------------------- device kind
    def test_parse_bench(self):
        b = ab_report.parse_bench(bench_text(64, 12.5, att=3.25))
        self.assertEqual(b["C"], 64)
        self.assertEqual(b["full_steps"], 38)
        self.assertAlmostEqual(b["device_ms_per_pass"], 12.5)
        self.assertAlmostEqual(b["stage_att_core"], 3.25)
        self.assertEqual(len([k for k in b if k.startswith("stage_")]), 14)
        self.assertIsNone(ab_report.parse_bench("no bench here\n"))

    def test_device_ab(self):
        s = Session(self.root, "dev", kind="device", levels=(64, 128))
        ms = {"A": [12.0, 12.1], "B": [10.0, 10.05]}
        d = s.write(lambda r: bench_text(r["level"], ms[r["arm"]][nth(r) if r["level"] == 64
                                                                  else (0 if r["seq"] <= 6 else 1)],
                                         att=2.0 if r["arm"] == "B" else 4.0))
        rc, v, txt = run(d)
        self.assertEqual(rc, 0)
        self.assertEqual(v["primary"], ["device_ms_per_pass"])
        self.assertEqual(v["verdicts"]["B"], "PASS")
        self.assertEqual(v["levels_report"]["64"]["compare"]["B"]["stage_att_core"]["verdict"],
                         "BETTER")
        self.assertIn("not applicable", v["gates"]["identity"]["note"])

    def test_device_failures(self):
        s = Session(self.root, "devbad", kind="device")
        d = s.write(lambda r: "FAIL open: no device\n" if r["arm"] == "B" else bench_text(64, 12.0),
                    meta=lambda r: {"rc": 1} if r["arm"] == "B" else None)
        rc, v, _ = run(d)
        self.assertEqual(rc, 2)
        probs = " | ".join(v["gates"]["integrity"]["problems"])
        self.assertIn("bench exit 1", probs)
        self.assertIn("no BENCH C= line", probs)

    def test_device_zero_steps_and_wrong_c(self):
        s = Session(self.root, "devzero", kind="device")
        d = s.write(lambda r: bench_text(32, 12.0).replace("full_steps=38 passes=38",
                                                            "full_steps=0 passes=0"))
        rc, v, _ = run(d)
        probs = " | ".join(v["gates"]["integrity"]["problems"])
        self.assertEqual(rc, 2)
        self.assertIn("nothing measured", probs)
        self.assertIn("BENCH C=32 but the run asked for C=16", probs)

    def test_single_pair_is_null(self):
        s = Session(self.root, "single", order="AB")
        lag = {"A": [100], "B": [50]}
        d = s.write(lambda r: load_json(16, 2, lag[r["arm"]][0], 300, 900))
        _, v, _ = run(d)
        c = v["levels_report"]["16"]["compare"]["B"]["lag_p95"]
        self.assertEqual(c["verdict"], "NULL")
        self.assertIn("noise floor", c["note"])

    def test_directory_of_sessions(self):
        for tag in ("s1", "s2"):
            Session(self.root, tag).write(lambda r: load_json(16, 2, 100, 300, 900))
        out = io.StringIO()
        old = sys.stdout
        sys.stdout = out
        try:
            rc = ab_report.main([self.root])
        finally:
            sys.stdout = old
        self.assertEqual(rc, 0)
        self.assertEqual(out.getvalue().count("== ab_report"), 2)


if __name__ == "__main__":
    unittest.main()
