#!/usr/bin/env python3
"""CPU self-test of the finetune/ helpers: no GPU, no NeMo, no network.

    python3 finetune/tests/test_finetune.py        # last line: "finetune self-test: PASS (...)"

Covers manifest rows/round trip and the voiced-span trim, nested + deterministic
subset selection, replay mixing ratio, run.json serialisation, the EOU/EOB/blank
id contract, a tiny SentencePiece round trip with <EOU>/<EOB> appended (skipped
with a message when sentencepiece is not installed), and the plain-ASR CLI presets.
"""
from __future__ import annotations

import io
import json
import os
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "common"), str(ROOT / "eou"), str(ROOT / "canary")]

import ftlib  # noqa: E402
import runmeta  # noqa: E402
import subsets  # noqa: E402
import tokenizer_eou  # noqa: E402
import plain_asr  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None
try:
    import sentencepiece  # noqa: F401
    HAVE_SPM = True
except ImportError:  # pragma: no cover
    HAVE_SPM = False


def synth_groups(n_spk=12, per=40, seed=3):
    rng = random.Random(seed)
    return {f"s{s:02d}": [{"id": f"s{s:02d}_{i:03d}", "speaker": f"s{s:02d}", "duration": rng.uniform(5, 20)}
                          for i in range(per)] for s in range(n_spk)}


class Manifests(unittest.TestCase):
    def test_canary_row_fields(self):
        r = ftlib.canary_row("/a/b.wav", 3.14159, "Perché sì", "it", "yes", corpus="fleurs", utt_id="u1")
        self.assertEqual(r["duration"], 3.142)
        self.assertEqual((r["source_lang"], r["target_lang"], r["pnc"], r["taskname"]), ("it", "it", "yes", "asr"))
        self.assertEqual(r["sampling_rate"], 16000)
        self.assertEqual(r["corpus"], "fleurs")

    def test_round_trip_unicode(self):
        rows = [ftlib.canary_row(f"/x/{i}.wav", 1 + i, "città è più già", "it", "no") for i in range(5)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sub" / "m.json"
            ftlib.write_manifest(p, rows)
            self.assertEqual(ftlib.read_manifest(p), rows)
            self.assertIn("città", p.read_text(encoding="utf-8"))  # ensure_ascii=False
            self.assertFalse(p.with_suffix(".json.tmp").exists())

    @unittest.skipIf(np is None, "numpy not installed")
    def test_voiced_span_trim(self):
        import trim

        sr = 16000
        a = np.zeros(sr * 3, dtype=np.float32)
        t = np.arange(sr) / sr
        a[sr: 2 * sr] = 0.3 * np.sin(2 * np.pi * 220 * t)  # speech stand-in from 1.0 s to 2.0 s
        off, dur = trim.voiced_span_of(a, sr)
        self.assertAlmostEqual(off, 0.9, delta=0.03)
        self.assertAlmostEqual(off + dur, 2.1, delta=0.03)

    def test_plain_norm_and_eou_norm(self):
        self.assertEqual(plain_asr.norm("Perché È COSÌ, l'acqua!"), "perche e cosi l'acqua")
        self.assertEqual(tokenizer_eou.norm_text("Perché È così, l’acqua 3!"), "perché è così l'acqua")


class Subsets(unittest.TestCase):
    def test_deterministic(self):
        g = synth_groups()
        a = [u["id"] for u in subsets.balanced_order(g, 600, seed=7)]
        b = [u["id"] for u in subsets.balanced_order(synth_groups(), 600, seed=7)]
        c = [u["id"] for u in subsets.balanced_order(synth_groups(), 600, seed=8)]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_nested_and_capped(self):
        seq = subsets.balanced_order(synth_groups(), 600, seed=7)
        cuts = subsets.nested_cuts(seq, "0.5,1,2")
        self.assertEqual(list(cuts), ["0.5h", "1h", "2h"])
        n = list(cuts.values())
        self.assertTrue(n[0] <= n[1] <= n[2])
        for h, k in cuts.items():
            s = sum(u["duration"] for u in seq[:k])
            self.assertLessEqual(s, float(h[:-1]) * 3600 + 1e-6)
        ids = [u["id"] for u in seq]
        self.assertEqual(len(ids), len(set(ids)))
        per_spk = {}
        for u in seq:
            per_spk[u["speaker"]] = per_spk.get(u["speaker"], 0) + u["duration"]
        self.assertTrue(all(v < 600 + 20 for v in per_spk.values()))  # cap overshoot <= one utterance

    def test_merge_by_share(self):
        mk = lambda tag, n: [{"id": f"{tag}{i}", "duration": 10.0} for i in range(n)]  # noqa: E731
        out = subsets.merge_by_share({"a": mk("a", 100), "b": mk("b", 100)}, {"a": 0.8, "b": 0.2})
        first = out[:50]
        share_b = sum(u["id"].startswith("b") for u in first) / len(first)
        self.assertAlmostEqual(share_b, 0.2, delta=0.05)
        self.assertEqual(len(out), 200)


class Replay(unittest.TestCase):
    def rows(self):
        it = [{"duration": 10.0, "target_lang": "it", "id": f"it{i}"} for i in range(360)]  # 1 h
        rp = [{"duration": 5.0, "target_lang": lg, "id": f"{lg}{i}"} for lg in ("en", "de") for i in range(100)]
        return it, rp

    def test_ratio_and_determinism(self):
        it, rp = self.rows()
        rows, info = ftlib.mix_replay(it, rp, 0.2, seed=1)
        self.assertAlmostEqual(info["replay_share"], 0.2, delta=0.002)
        self.assertEqual(len(rows), len(it) + info["replay_rows"])
        self.assertEqual(set(info["replay_s_per_lang"]), {"en", "de"})
        rows2, _ = ftlib.mix_replay(it, rp, 0.2, seed=1)
        self.assertEqual([r["id"] for r in rows], [r["id"] for r in rows2])
        self.assertNotEqual([r["id"] for r in rows[:360]], [r["id"] for r in it])  # shuffled as a whole

    def test_pool_repeats_when_short(self):
        it, rp = self.rows()
        _, info = ftlib.mix_replay(it, rp[:10], 0.5, seed=1)
        self.assertGreater(info["pool_passes"], 1)
        self.assertAlmostEqual(info["replay_share"], 0.5, delta=0.01)

    def test_bad_ratio(self):
        it, rp = self.rows()
        with self.assertRaises(AssertionError):
            ftlib.mix_replay(it, rp, 1.0, seed=1)


class RunJson(unittest.TestCase):
    def test_build_write_read(self):
        with tempfile.TemporaryDirectory() as d:
            tok = Path(d) / "tok"
            tok.mkdir()
            (tok / "tokenizer.model").write_bytes(b"abc")
            run = runmeta.build("finetune/x.py", {"lr": 2e-4, "out": Path(d)}, probe_env=False,
                                base_model={"id": "nvidia/x", "revision": None, "path": Path(d) / "m.nemo", "sha256": None},
                                tokenizer={"path": tok, "sha256": runmeta.sha256_path(tok)},
                                lr_groups={"default": 2e-4, "encoder": 6e-5}, max_steps=10, fastemit_lambda=0.0)
            p = Path(d) / "run" / "run.json"
            runmeta.write(p, run, status="started")
            back = json.loads(p.read_text())
            self.assertEqual(back["schema"], runmeta.SCHEMA)
            self.assertEqual(back["status"], "started")
            self.assertEqual(back["config"]["out"], d)  # Path -> str
            self.assertTrue(back["tokenizer"]["sha256"].startswith("dir:"))
            for k in ("git", "seed", "optimizer", "scheduler", "augmentation", "eou_plain_mix", "freeze_policy",
                      "hardware", "versions", "outputs"):
                self.assertIn(k, back)
            runmeta.write(p, back, status="done", outputs={"final": Path(d) / "final.nemo"})
            self.assertEqual(json.loads(p.read_text())["status"], "done")
        with self.assertRaises(AssertionError):
            runmeta.build("x", {}, probe_env=False, not_a_field=1)

    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a"
            f.write_bytes(b"abc")
            self.assertEqual(runmeta.sha256_path(f),
                             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
            self.assertIsNone(runmeta.sha256_path(Path(d) / "missing"))

    def test_require_root(self):
        old = os.environ.pop("FT_ROOT", None)
        try:
            with self.assertRaises(SystemExit):
                runmeta.require_root()
            os.environ["FT_ROOT"] = "/tmp/x"
            self.assertEqual(str(runmeta.require_root()), "/tmp/x")
        finally:
            os.environ.pop("FT_ROOT", None)
            if old is not None:
                os.environ["FT_ROOT"] = old


class EouIds(unittest.TestCase):
    def test_contract(self):
        self.assertEqual(tokenizer_eou.check_special_ids(1026, 1024, 1025, base_vocab=1024),
                         {"vocab": 1026, "eou": 1024, "eob": 1025, "blank": 1026})
        for bad in ((1026, 1025, 1024), (1026, 0, 1025), (1027, 1024, 1025)):
            with self.assertRaises(AssertionError):
                tokenizer_eou.check_special_ids(*bad)
        with self.assertRaises(AssertionError):
            tokenizer_eou.check_special_ids(1026, 1024, 1025, blank_id=0)
        with self.assertRaises(AssertionError):
            tokenizer_eou.check_special_ids(1026, 1024, 1025, base_vocab=1000)

    @unittest.skipUnless(HAVE_SPM, "sentencepiece not installed: tiny SPE round trip skipped")
    def test_tiny_spe_round_trip(self):
        spec = {"model_type": "bpe", "byte_fallback": False, "split_digits": False, "unk_id": 0, "bos_id": -1,
                "eos_id": -1, "pad_id": -1, "normalization_rule_name": "nmt_nfkc"}
        words = "la casa è bella l'acqua scorre nel fiume perché così città più già ciao come stai".split()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            txt = d / "pool.txt"
            txt.write_text("\n".join(" ".join(words[(i * 7 + j) % len(words)] for j in range(12))
                                     for i in range(300)) + "\n", encoding="utf-8")
            sink = io.StringIO()
            with redirect_stdout(sink), redirect_stderr(sink):
                base = tokenizer_eou.train_spm(txt, d, spec, 80)
                tokenizer_eou.add_special(base, d)
                res = tokenizer_eou.verify(d, 80, nemo_check=False)
            self.assertEqual((res["eou_id"], res["eob_id"], res["blank_id_in_model"]), (80, 81, 82))
            import sentencepiece as spm

            sp = spm.SentencePieceProcessor(model_file=str(d / "tokenizer.model"))
            for s in ("la casa è bella", "perché così città<EOU>"):
                self.assertEqual(sp.decode(sp.encode(s)), s)
            # the base pieces keep their ids after the append
            sp0 = spm.SentencePieceProcessor(model_file=str(base))
            self.assertEqual([sp0.id_to_piece(i) for i in range(80)], [sp.id_to_piece(i) for i in range(80)])
            self.assertIn("<EOU>", (d / "tokenizer.vocab").read_text(encoding="utf-8"))
            self.assertNotIn("<EOU>", (d / "vocab.txt").read_text(encoding="utf-8"))


class PlainCli(unittest.TestCase):
    base = ["--manifest", "m.json", "--out", "runs", "--tag", "t"]

    def test_plain_defaults(self):
        a = plain_asr.parse_args(self.base)
        self.assertEqual((a.subset_utts, a.max_steps, a.lr, a.freeze_enc, a.train_eval_n), (0, 400, 3e-4, 1, 8))
        self.assertEqual(plain_asr.eval_steps_of(a), [100, 200, 300, 400])
        self.assertEqual(a.fastemit, 0.0)

    def test_micro_overfit_preset_and_override(self):
        a = plain_asr.parse_args(self.base + ["--micro-overfit", "--lr", "2e-4"])
        self.assertEqual((a.subset_utts, a.max_steps, a.lr, a.train_eval_n), (32, 1000, 2e-4, 32))
        self.assertEqual(plain_asr.eval_steps_of(a), [100, 250, 500, 750, 1000])
        a = plain_asr.parse_args(self.base + ["--n", "16", "--steps", "300", "--eval-steps", "50,100"])
        self.assertEqual((a.subset_utts, a.max_steps), (16, 300))
        self.assertEqual(plain_asr.eval_steps_of(a), [50, 100, 300])

    def test_requires_output(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            plain_asr.parse_args(["--manifest", "m.json", "--tag", "t"])

    def test_wer_cer(self):
        w, c = plain_asr.wer_cer(["ciao come stai"], ["ciao come"])
        self.assertAlmostEqual(w, 100 / 3, places=4)
        self.assertGreater(c, 0)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    res = unittest.TextTestRunner(verbosity=1, stream=sys.stderr).run(suite)
    skipped = len(res.skipped)
    for t, why in res.skipped:
        print(f"SKIP {t.id().split('.')[-1]}: {why}")
    ok = res.wasSuccessful()
    print(f"finetune self-test: {'PASS' if ok else 'FAIL'} ({res.testsRun} tests, {skipped} skipped)")
    sys.exit(0 if ok else 1)
