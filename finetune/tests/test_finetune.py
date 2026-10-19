#!/usr/bin/env python3
"""CPU self-test of the finetune/ helpers: no GPU, no NeMo, no network.

    python3 finetune/tests/test_finetune.py        # last line: "finetune self-test: PASS (...)"

Covers manifest rows/round trip and the voiced-span trim, nested + deterministic
subset selection, replay mixing ratio, run.json serialisation, the EOU/EOB/blank
id contract, a tiny SentencePiece round trip with <EOU>/<EOB> appended (skipped
with a message when sentencepiece is not installed), the plain-ASR CLI presets, and the shared
stock-SPE text normaliser (common/textnorm.py: IT == the frozen scorer rule, FR oe/ae + typographic apostrophes),
and the --lang fr wiring (FR-P1: sha1 / equal-share MLS order, prepare_it fr plan, prepare_mix text_ok, teacher_wer,
two_stage / lm_fusion_probe norm, the a2_vpfilter threshold).
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


class TextNorm(unittest.TestCase):
    """common/textnorm.py: lang="it" is the frozen scorer rule, lang="fr" adds oe/ae + typographic apostrophes."""
    IT = ["Perché È COSÌ, l'acqua!", "dell’anno «più» bello", "c'è-un po' di città 3 volte", "  ŒUVRE  ", "Ærøskøbing",
          "‘ciao’ ʼx", "ﬁne ¼ Ⅻ ß ı İ ø ł đ", "é́ é", "a\tb\nc", ""]

    def test_it_is_the_frozen_rule(self):
        import textnorm
        for t in self.IT + [chr(c) for c in range(0x20, 0x3000) if not 0xD800 <= c <= 0xDFFF]:
            self.assertEqual(textnorm.norm(t, "it"), plain_asr.norm(t), repr(t))   # plain_asr.norm == two_stage.norm
        self.assertEqual(textnorm.norm("dell’anno", "it"), "dell anno")   # the IT word break on ’ stays frozen

    def test_fr(self):
        import textnorm
        cases = {"L’œuvre d’art, peut-être.": "l'oeuvre d'art peut etre",
                 "Cœur, ŒUVRE, Lætitia, Æsope": "coeur oeuvre laetitia aesope",
                 "aujourd’hui c‘est ʼla": "aujourd'hui c'est 'la", "c'est-à-dire « oui » — non": "c'est a dire oui non",
                 "Noël à l'hôpital, où ça ?": "noel a l'hopital ou ca", "Il a 3 ans": "il a ans"}
        for t, want in cases.items():
            self.assertEqual(textnorm.norm(t, "fr"), want, t)
        for t in ("Perché È COSÌ, l'acqua!", "c'è-un po' di città"):   # no ligature / curly apostrophe: same as IT
            self.assertEqual(textnorm.norm(t, "fr"), textnorm.norm(t, "it"))
        with self.assertRaises(KeyError):
            textnorm.norm("x", "xx")

    def test_text_ok(self):
        import textnorm
        self.assertFalse(textnorm.text_ok("le cœur", "it"))   # prepare_mix.text_ok drops every oe / ae row ...
        self.assertTrue(textnorm.text_ok("le cœur", "fr"))    # ... the FR filter keeps them
        for t, ok in (("città è più", True), ("l’acqua", True), ("3 volte", False), ("Москва", False), ("", False),
                      ("ñandú", True), ("straße", False), ("...", False)):
            self.assertEqual((textnorm.text_ok(t, "it"), textnorm.text_ok(t, "fr")), (ok, ok), t)


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


class Sha1Order(unittest.TestCase):
    """FR-P1 MLS order: seed-free (sha1), least-duration-first, equal-share cap over every speaker."""

    def test_deterministic_nested_spread(self):
        g = synth_groups(n_spk=30, per=60)
        a = subsets.balanced_order_sha1(g, 1e9)
        self.assertEqual([u["id"] for u in a], [u["id"] for u in subsets.balanced_order_sha1(synth_groups(30, 60), 1e9)])
        self.assertNotEqual([u["id"] for u in a], [u["id"] for u in subsets.balanced_order_sha1(g, 1e9, salt="x")])
        cap = subsets.equal_share_cap_min(g, 2)
        self.assertEqual(cap, 4)   # ceil(2 h / 30 speakers = 4.0 min)
        seq = subsets.balanced_order_sha1(g, cap * 60)
        k = subsets.nested_cuts(seq, "2")["2h"]
        per = {}
        for u in seq[:k]:
            per[u["speaker"]] = per.get(u["speaker"], 0) + u["duration"]
        self.assertEqual(len(per), 30)                                   # every speaker is in the cut
        self.assertLess(max(per.values()) - min(per.values()), 20 + 1e-6)  # balanced to within one clip

    def test_equal_share_cap(self):
        mk = lambda mins: {f"s{i}": [{"id": f"s{i}", "duration": m * 60.0}] for i, m in enumerate(mins)}  # noqa: E731
        self.assertEqual(subsets.equal_share_cap_min(mk([100] * 142), 40), 17)   # 2400 / 142 = 16.9 -> 17
        self.assertEqual(subsets.equal_share_cap_min(mk([1, 1, 100, 100]), 1), 29)   # (60 - 2) / 2 = 29
        self.assertEqual(subsets.equal_share_cap_min(mk([5, 7]), 1), 7)   # pool shorter than the cut


class LangWiring(unittest.TestCase):
    """--lang fr paths of the data / training / probe scripts; lang it (default) unchanged."""

    def plan(self, lang, edit=None):
        import argparse
        import prepare_it as D
        old_ft, old_meta = D.FT, D.synthetic_meta
        try:
            with tempfile.TemporaryDirectory() as d:
                D.FT = Path(d)
                D.set_lang(lang)
                if edit:
                    D.synthetic_meta = lambda: edit(old_meta())
                a = argparse.Namespace(seed=20261009, hours="5,20,40", min_dur=1.0, max_dur=20.0, mls_cap_min=None,
                                       mls_cap_rule_auto=True, with_fleurs_train=False, fleurs_share=0.15,
                                       fleurs_per_sentence=2, en_clips=100)
                with redirect_stdout(io.StringIO()):
                    p = D.stage_plan(a, synthetic=True)
                return p, sorted(os.listdir(d)), D.tag("mls"), D.MLS_TRAIN, D.FL
        finally:
            D.FT, D.synthetic_meta = old_ft, old_meta
            D.set_lang("it")

    def test_prepare_it_fr_plan(self):
        p, files, mls_dir, shards, fl = self.plan("fr")
        self.assertEqual(files, ["plan_fr.synthetic.json"])
        self.assertEqual((mls_dir, len(shards), shards[-1], fl), ("mls_fr", 34, "french/train-00033-of-00034.parquet", "fr_fr"))
        self.assertEqual(sorted(p["eval"]), ["fleurs_fr_test", "mls_fr_test"])
        self.assertEqual(p["en_sanity"], [])
        self.assertEqual((p["mls_order"], p["lang"]), ("sha1", "fr"))
        c = p["cuts"]["40h"]
        self.assertEqual(c["mls_speakers"], p["pool"]["mls_speakers"])   # the 40 h cut touches every pool speaker
        self.assertEqual(p["mls_cap_min"], 37)   # synthetic: 65 speakers, 40 h -> equal share 36.9 min, rounded up
        self.assertLessEqual(c["mls_min_per_speaker_max"], p["mls_cap_min"] + 20 / 60 + 1e-6)
        self.assertEqual(len(c["mls_clips_per_speaker_min_med_max"]), 3)

    def test_prepare_it_fr_text_filter(self):
        def edit(m):
            for r in m[0][:50]:
                r["transcript"] = "il a 3 ans"
            return m
        p, *_ = self.plan("fr", edit)
        self.assertEqual(p["pool"]["dropped"]["text"], 50)
        self.assertFalse(any("3" in u["text"] for u in p["train_sequence"]))
        q, files, mls_dir, shards, fl = self.plan("it", edit)   # the IT pool keeps them (frozen IT plan)
        self.assertEqual(q["pool"]["dropped"], {"duration": 0, "test_speaker": 0})
        self.assertEqual((files, mls_dir, len(shards), fl), (["plan.synthetic.json"], "mls", 8, "it_it"))
        self.assertEqual(sorted(q["eval"]), ["fleurs_it_test", "mls_it_test"])
        self.assertEqual((q["mls_cap_min"], len(q["en_sanity"])), (60.0, 100))
        self.assertNotIn("mls_order", q)

    def test_prepare_mix_text_ok(self):
        import prepare_mix
        for t in ("città è più", "l’acqua", "3 volte", "le cœur", "Москва", ""):
            self.assertEqual(prepare_mix.text_ok(t), prepare_mix.textnorm.text_ok(t, "it"), t)
        self.assertEqual((prepare_mix.text_ok("le cœur"), prepare_mix.text_ok("le cœur", "fr")), (False, True))

    def test_teacher_wer_rule(self):
        sys.path.insert(0, str(ROOT / "eou" / "diag"))
        import teacher_ws
        self.assertEqual(teacher_ws.teacher_wer("a b c d", ""), 100.0)     # empty teacher -> 100 (dropped at t50)
        self.assertIsNone(teacher_ws.teacher_wer("", "a b"))               # no reference -> None (dropped)
        self.assertEqual(teacher_ws.teacher_wer("a b c", "a x c"), 33.3)
        self.assertEqual(teacher_ws.teacher_wer("a b", "a b c d"), 100.0)
        old = teacher_ws.NORM_LANG
        try:
            teacher_ws.NORM_LANG = "fr"
            self.assertEqual(teacher_ws.norm("L’œuvre"), "l'oeuvre")
            teacher_ws.NORM_LANG = "it"
            self.assertEqual(teacher_ws.norm("L’œuvre"), "l uvre")   # the frozen IT rule
        finally:
            teacher_ws.NORM_LANG = old

    def test_script_norm_follows_lang(self):
        """two_stage.py / lm_fusion_probe.py run at import (NeMo): check their --lang default and norm() by AST."""
        import argparse
        import ast
        import textnorm
        for rel in ("eou/two_stage.py", "eou/diag/lm_fusion_probe.py"):
            tree = ast.parse((ROOT / rel).read_text())
            args = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"
                    and n.args and getattr(n.args[0], "value", None) == "--lang"]
            self.assertEqual(len(args), 1, rel)
            self.assertEqual({k.arg: getattr(k.value, "value", None) for k in args[0].keywords}["default"], "it", rel)
            fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "norm"]
            for lang, want in (("it", "l uvre d art"), ("fr", "l'oeuvre d'art")):
                ns = {"textnorm": textnorm, "a": argparse.Namespace(lang=lang)}
                exec(compile(ast.Module(fn, []), rel, "exec"), ns)
                self.assertEqual(ns["norm"]("L’œuvre d’art"), want, (rel, lang))

    def test_vp_filter_threshold(self):
        """a2_vpfilter.sh's inline filter: keep iff teacher_wer is not None and <= THR (inclusive)."""
        import subprocess
        src = (ROOT / "jobs" / "a2_vpfilter.sh").read_text()
        code = src.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        rows = [{"audio_filepath": f"/a/{i}.wav", "duration": 2.0, "speaker": "s", "text": "x", "teacher": "x",
                 "teacher_wer": w} for i, w in enumerate((0.0, 50.0, 50.1, 100.0, None))]
        with tempfile.TemporaryDirectory() as d:
            inp, out = Path(d) / "t.json", Path(d) / "k.json"
            ftlib.write_manifest(inp, rows)
            r = subprocess.run([sys.executable, "-", str(inp), str(out), "50"], input=code, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            kept = ftlib.read_manifest(out)
        self.assertEqual([k["audio_filepath"] for k in kept], ["/a/0.wav", "/a/1.wav"])
        self.assertNotIn("teacher_wer", kept[0])


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


class FinalLangBuild(unittest.TestCase):
    """eou/prepare_final_it.py --lang it|fr on an offline fake Hub (no network): the IT sources / paths / rules stay
    the frozen FINAL-IT ones, the FR wiring (paths, listed mTEDx shards, Granary-fr, Speech-MASSIVE, textnorm fr),
    and --plan-only (metadata only, nothing sealed). world() / run() are reused by the one-off IT identity check
    (old vs new module, byte-compared outputs)."""
    HF = "https://huggingface.co/datasets"

    @classmethod
    def setUpClass(cls):
        try:
            import soundfile  # noqa: F401
            import prepare_final_it
        except ImportError as e:  # pragma: no cover
            raise unittest.SkipTest(f"prepare_final_it deps missing: {e}")
        cls.mod = prepare_final_it

    @staticmethod
    def wav(dur, tail=0.4):
        import soundfile as sf
        n, k = int(dur * 16000), int((dur - tail) * 16000)
        x = np.zeros(n, dtype=np.float32)
        x[:k] = 0.3 * np.sin(2 * np.pi * 200 * np.arange(k) / 16000)
        b = io.BytesIO()
        sf.write(b, x, 16000, format="WAV", subtype="PCM_16")
        return b.getvalue()

    @classmethod
    def world(cls, lang):
        """{(repo, path): rows}, tsv files, tars, zips and tree listings of a tiny synthetic Hub for one language."""
        w = cls.wav
        L, mlsd, fl = {"it": ("it", "italian", "it_it"), "fr": ("fr", "french", "fr_fr")}[lang]
        import collections
        P, T = {}, collections.defaultdict(list)   # parquet rows, tree listings

        def add(repo, path, rows):
            P[(repo, path)] = rows
            d, f = path.rsplit("/", 1)
            T[(repo, d)].append({"type": "file", "path": path})
            assert f
        texts = ["la casa è bella e grande davvero", "l’anno scorso è stato lungo", "le cœur de la ville",
                 "il a 3 ans", "oui", "testo già usato nel training", "c'è-un po' di città qui", "dup tra banche ok"]
        R = "facebook/multilingual_librispeech"
        add(R, f"{mlsd}/train-00000-of-00002.parquet", [{"speaker_id": 1}])
        add(R, f"{mlsd}/train-00001-of-00002.parquet", [{"speaker_id": 2}])
        add(R, f"{mlsd}/test-00000-of-00001.parquet", [{"speaker_id": 3}])
        add(R, f"{mlsd}/1_hours-00000-of-00001.parquet", [{"speaker_id": 10}])   # not train: never read
        add(R, f"{mlsd}/dev-00000-of-00001.parquet",
            [{"id": f"m{i}", "speaker_id": 10 + i % 3, "transcript": t, "audio_duration": d, "audio": {"bytes": w(d)}}
             for i, (t, d) in enumerate(zip(texts + ["troppo corto davvero", "troppo lungo davvero"],
                                            [3, 3, 3, 3, 3, 3, 3, 3, 0.5, 21]))])
        R = "fixie-ai/common_voice_17_0"
        add(R, f"{L}/train-00000-of-00001.parquet", [{"client_id": "ctrain"}])
        add(R, f"{L}/test-00000-of-00001.parquet", [{"client_id": "ctest"}])
        for s in (0, 1):
            add(R, f"{L}/validation-0000{s}-of-00002.parquet",
                [{"client_id": c, "path": f"/x/cv_{s}_{i}.mp3", "sentence": t, "audio": {"bytes": w(d)}}
                 for i, (c, t, d) in enumerate([("ca", "dup tra banche ok", 2.5), ("ctrain", "frase del treno", 2),
                                                ("cb", "Une phrase « entre guillemets ».", 2.2), ("cc", "lunga", 2),
                                                ("cd", f"frase numero {'due' if s else 'uno'} pulita", 3)])])
        R = "facebook/voxpopuli"
        add(R, f"{L}/train-00000-of-00001.parquet", [{"speaker_id": "v1"}, {"speaker_id": "None"}])
        add(R, f"{L}/test-00000-of-00001.parquet", [{"speaker_id": "v2"}])
        add(R, f"{L}/validation-00000-of-00001.parquet",
            [{"audio_id": f"vp{i}", "speaker_id": s, "normalized_text": t, "audio": {"bytes": w(d)}}
             for i, (s, t, d) in enumerate([("v1", "speaker del training", 3), ("v3", "il parlamento europeo decide.", 4),
                                            ("None", "nessun oratore noto qui.", 3), ("v4", "un altro intervento lungo", 25)])])
        tsv = {sp: [[str(100 + i + k), f"{sp}{i}.wav", t, t.lower(), "1", str(int(d * 16000)), "MALE"]
                    for i, (t, d) in enumerate(rows)]
               for k, (sp, rows) in enumerate([("train", [("frase uno", 3)]), ("test", [("frase due", 3)]),
                                               ("dev", [("la frase di sviluppo uno", 3), ("la frase di sviluppo due", 4),
                                                        ("frase senza audio qui", 3)])])}
        tsv["train"][0][0], tsv["test"][0][0] = "1", "2"
        tars = {f"{cls.HF}/google/fleurs/resolve/main/data/{fl}/audio/dev.tar.gz":
                {f"dev/dev{i}.wav": w(d) for i, d in ((0, 3), (1, 4))}}
        R = "espnet/yodas-granary"
        for d in (f"{L}000", f"{L}100", "en000", "itx00" if lang == "it" else "frx00"):
            T[(R, "data")].append({"type": "directory", "path": f"data/{d}"})
        add(R, f"data/{L}000/asr_only/a.parquet", [{"original_audio_id": "GRANARYTALK"}])
        add(R, f"data/{L}100/ast/b.parquet", [{"original_audio_id": "yt:GRANARYTAL2"}])
        add(R, "data/en000/asr_only/e.parquet", [{"original_audio_id": "ENTALK000001"}])
        R = "deepdml/mtedx"
        add(R, f"{L}/test-00000-of-00001.parquet", [{"talk_id": "devtalk"}])
        add(R, f"{L}/valid-00000-of-00001.parquet", [{"talk_id": "devtalk2"}])
        nsh = 30 if lang == "it" else 41
        segs = [("t1", "Bonjour à tous, merci. »", 3), ("t1", "Vous le savez ?", 2), ("t1", "una frase senza fine", 3),
                ("devtalk", "un talk del dev bank.", 3), ("GRANARYTALK", "un talk visto in Granary.", 3),
                ("GRANARYTAL2", "anche questo in Granary.", 3), ("t2", "Ecco due parole.", 2), ("t2", "è finita così!", 2.5),
                ("ENTALK000001", "un talk inglese non escluso.", 3)]
        for s in range(nsh):
            rows = [{"id": f"{tk}_{s}_{i}", "talk_id": tk, "transcript": t, "duration": d, "audio": {"bytes": w(d)}}
                    for i, (tk, t, d) in enumerate(segs)] if s < 2 else [{"id": "never", "talk_id": "z"}]
            add(R, f"{L}/train-{s:05d}-of-{nsh:05d}.parquet", rows)
        R = "FBK-MT/Speech-MASSIVE"
        add(R, "fr-FR/train-00000-of-00001.parquet", [{"speaker_id": "strain"}])
        add(R, "fr-FR/train_115-00000-of-00001.parquet", [{"speaker_id": "s115"}])
        for s in (0, 1):
            add(R, f"fr-FR/validation-0000{s}-of-00002.parquet",
                [{"path": f"dev/h{s}{i}.wav", "speaker_id": sp, "utt": t, "is_validated": ok, "audio": {"bytes": w(d)}}
                 for i, (sp, t, ok, d) in enumerate([("sa", ["réveille moi à sept heures", "réveille moi demain"][s], True, 2),
                                                     ("strain", "quel temps fait il", True, 2),
                                                     ("s115", "allume la lumière", True, 2),
                                                     ("sb", "joue de la musique douce", False, 2),
                                                     ("sb" if s else "sc", ["mets un minuteur cœur", "baisse le son"][s],
                                                      True, 2)])])
        items = [{"id": 7 + i, "speaker_id": sp, "utt": t, "duration": 2.0} for i, (sp, t) in
                 enumerate([("i1", "accendi la luce in cucina"), ("i2", "spegni la radio adesso"), ("i1", "ok")])]
        zb = io.BytesIO()
        import zipfile
        with zipfile.ZipFile(zb, "w") as z:
            z.writestr("zenodo_dataset/hard_speaker_validation.json", "\n".join(json.dumps(r) for r in items))
            z.writestr("zenodo_dataset/hard_speaker_test.json", json.dumps({"id": 1, "speaker_id": "i9", "utt": "x"}))
            for r in items:
                z.writestr(f"zenodo_dataset/recordings/{r['id']}.wav", w(2.0))
        return {"P": P, "T": dict(T), "tsv": tsv, "tars": tars, "zip": zb.getvalue(), "fleurs": fl}

    @classmethod
    def drive(cls, mod, d, argv, world):
        """run mod.main() on the fake Hub; returns the request log (sorted: Hub.columns reads in threads)."""
        import tarfile
        from unittest import mock
        log, reg = [], {}
        W = world

        class FakeHub(mod.Hub):
            def __init__(self):
                self.tok, self.cap, self.block = None, 0, 0

            def tree(self, repo, p):
                log.append(("tree", repo, p))
                if repo == "FBK-MT/Speech-MASSIVE-test":
                    raise OSError("401 gated")
                if (repo, p) not in W["T"]:
                    raise OSError(f"404 {repo}/{p}")
                return W["T"][(repo, p)]

            def column(self, repo, path, cols):
                log.append(("column", repo, path, tuple(cols)))
                return [{c: r[c] for c in cols} for r in W["P"][(repo, path)]]

        def fetch(url, dst, tries=20):
            log.append(("fetch", url, os.path.basename(dst)))
            if url.endswith(".tsv"):
                sp = url.rsplit("/", 1)[-1][:-4]
                Path(dst).write_text("".join("\t".join(r) + "\n" for r in W["tsv"][sp]), encoding="utf-8")
            elif url.endswith(".zip"):
                Path(dst).write_bytes(W["zip"])
            else:
                reg[dst] = tuple(url[len(cls.HF) + 1:].split("/resolve/main/"))
            return dst

        def urlopen(url, timeout=None):
            log.append(("urlopen", url))
            b = io.BytesIO()
            with tarfile.open(fileobj=b, mode="w:gz") as t:
                for n, data in W["tars"][url].items():
                    ti = tarfile.TarInfo(n)
                    ti.size = len(data)
                    t.addfile(ti, io.BytesIO(data))
            b.seek(0)
            return b

        with mock.patch.object(mod, "Hub", FakeHub), mock.patch.object(mod, "fetch", fetch), \
                mock.patch.object(mod, "parquet_audio", lambda p: W["P"][reg[p]]), \
                mock.patch.object(mod.urllib.request, "urlopen", urlopen), \
                mock.patch.object(sys, "argv", ["prepare_final_it.py", "--out", f"{d}/out", "--tmp", f"{d}/tmp"] + argv), \
                redirect_stdout(io.StringIO()) as out:
            mod.main()
        return sorted(log, key=repr), out.getvalue()

    def build(self, lang, extra=()):
        d = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, d, True)
        Path(f"{d}/ex.json").write_text(json.dumps({"text": "Testo già usato nel training!"}) + "\n" +
                                        json.dumps({"text": "Le coeur de la ville"}) + "\n", encoding="utf-8")
        argv = ["--lang", lang, "--exclude-texts", f"{d}/ex.json"] + list(extra)
        if "--plan-only" not in extra:
            argv += ["--audio", f"{d}/audio"]
        log, out = self.drive(self.mod, d, argv, self.world(lang))
        rows = lambda b, k="final": [json.loads(x) for x in Path(f"{d}/out/{k}_{b}.json").read_text().splitlines()]  # noqa: E731,E501
        return d, log, out, rows

    def test_it_sources_frozen(self):
        d, log, out, rows = self.build("it")
        H = self.HF
        fetched = {e[1] for e in log if e[0] == "fetch"} | {e[1] for e in log if e[0] == "urlopen"}
        want = {f"{H}/facebook/multilingual_librispeech/resolve/main/italian/dev-00000-of-00001.parquet",
                f"{H}/fixie-ai/common_voice_17_0/resolve/main/it/validation-00000-of-00002.parquet",
                f"{H}/fixie-ai/common_voice_17_0/resolve/main/it/validation-00001-of-00002.parquet",
                f"{H}/facebook/voxpopuli/resolve/main/it/validation-00000-of-00001.parquet",
                *(f"{H}/google/fleurs/resolve/main/data/it_it/{s}.tsv" for s in ("train", "test", "dev")),
                f"{H}/google/fleurs/resolve/main/data/it_it/audio/dev.tar.gz",
                f"{H}/deepdml/mtedx/resolve/main/it/train-00000-of-00030.parquet",
                f"{H}/deepdml/mtedx/resolve/main/it/train-00001-of-00030.parquet",
                "https://zenodo.org/records/8040649/files/italic.zip"}
        self.assertEqual(fetched, want)
        self.assertNotIn(("tree", "deepdml/mtedx", "it"), log)   # the IT shard names are the frozen literals
        gran = {e[2] for e in log if e[0] == "column" and e[1] == "espnet/yodas-granary"}
        self.assertEqual(gran, {"data/it000/asr_only/a.parquet", "data/it100/ast/b.parquet"})
        self.assertIn("talk in Granary-it", out)
        self.assertIn("== FINAL-IT-DONE", out)
        mls = {r["utt_id"]: r for r in rows("mls")}
        # digit, short, too short / long, excluded text dropped; IT norm keeps the frozen word break on oe ("c ur")
        self.assertEqual(sorted(mls), ["m0", "m1", "m2", "m6", "m7"])
        self.assertEqual(mls["m1"]["text"], "l'anno scorso è stato lungo")
        # cross-bank dedup ("dup tra banche ok" taken by mls) and in-bank dedup (the same cb sentence in both shards)
        self.assertEqual(sorted(r["text"] for r in rows("cv")),
                         ["Une phrase « entre guillemets ».", "frase numero due pulita", "frase numero uno pulita"])
        self.assertEqual({r["utt_id"] for r in rows("vp")}, {"vp1"})   # no --vp-exclude: every VP train/test spk
        self.assertEqual(sorted(r["text"] for r in rows("tedx")),   # IT ending rule: "merci. »" (spaced) rejected
                         ["Ecco due parole.", "Vous le savez ?", "un talk inglese non escluso.", "è finita così!"])
        self.assertEqual({r["utt_id"] for r in rows("fleurs")}, {"dev0", "dev1"})   # dev2 has no audio
        self.assertEqual({r["utt_id"] for r in rows("italic")}, {7, 8})
        for b in ("mls", "tedx"):
            for r, t in zip(rows(b), rows(b, "final_trim")):
                self.assertTrue(Path(r["audio_filepath"]).exists() and Path(t["audio_filepath"]).exists())
                self.assertLess(t["duration"], r["duration"])
                self.assertAlmostEqual(r["trail_s"], 0.4, delta=0.03)
        sums = Path(f"{d}/out/SHA256SUMS").read_text().splitlines()
        self.assertEqual(len([s for s in sums if s.endswith(".wav")]), 2 * len(rows("all", "final")))

    def test_it_norm_and_ending_frozen(self):
        import re
        import textnorm
        m = self.mod
        self.assertIs(m.normf("it"), m.norm)
        for t in ("Perché È COSÌ, l'acqua!", "dell’anno «più»", "le cœur", "Ærø ʼx ‘y’", "c'è-un po' 3"):
            self.assertEqual(m.norm(t), textnorm.norm(t, "it"))
        self.assertEqual(m.normf("fr")("L’œuvre, peut-être."), "l'oeuvre peut etre")
        self.assertEqual(m.ENDING["it"], r"([.?!]|\.\.\.|…)[\"'»”]?$")
        ok = lambda lang, t: bool(re.search(m.ENDING[lang], t))  # noqa: E731
        for t, it, fr in (("fine.", 1, 1), ("davvero?", 1, 1), ("Vous savez ?", 1, 1), ("«ciao.»", 1, 1),
                          ("merci. »", 0, 1), ("e poi", 0, 0), ("allora…", 1, 1), ("così !\"", 1, 1), ("no »", 0, 0)):
            self.assertEqual((ok("it", t), ok("fr", t)), (bool(it), bool(fr)), t)
        self.assertEqual(m.SRC["it"]["banks"], "mls,cv,vp,fleurs,tedx,italic")
        self.assertEqual(m.SRC["fr"]["banks"], "mls,cv,vp,fleurs,tedx,smassive")

    def test_fr_build(self):
        d = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, d, True)
        Path(f"{d}/vp.json").write_text(json.dumps({"text": "x", "speaker": "vp:v3"}) + "\n", encoding="utf-8")
        d2, log, out, rows = self.build("fr", ["--vp-exclude", f"{d}/vp.json"])
        H = self.HF
        fetched = {e[1] for e in log if e[0] in ("fetch", "urlopen")}
        self.assertIn(f"{H}/facebook/multilingual_librispeech/resolve/main/french/dev-00000-of-00001.parquet", fetched)
        self.assertIn(f"{H}/facebook/voxpopuli/resolve/main/fr/validation-00000-of-00001.parquet", fetched)
        self.assertIn(f"{H}/google/fleurs/resolve/main/data/fr_fr/audio/dev.tar.gz", fetched)
        self.assertEqual({u for u in fetched if "mtedx" in u},   # listed (41 shards), not the IT literal
                         {f"{H}/deepdml/mtedx/resolve/main/fr/train-0000{i}-of-00041.parquet" for i in (0, 1)})
        self.assertFalse([u for u in fetched if "zenodo" in u or "/it/" in u or "italian" in u or "it_it" in u])
        gran = {e[2] for e in log if e[0] == "column" and e[1] == "espnet/yodas-granary"}
        self.assertEqual(gran, {"data/fr000/asr_only/a.parquet", "data/fr100/ast/b.parquet"})
        self.assertIn("== FINAL-FR-DONE", out)
        self.assertIn("GRANARYTAL2", json.loads(Path(f"{d2}/tmp/granary_fr_ids.json").read_text()))   # cached listing
        self.assertIn("Speech-MASSIVE-test not readable", out)
        self.assertEqual(sorted(p.name for p in Path(f"{d2}/out").glob("final_*.json") if "trim" not in p.name),
                         ["final_all.json", "final_cv.json", "final_fleurs.json", "final_mls.json",
                          "final_smassive.json", "final_tedx.json", "final_vp.json"])
        mls = {r["utt_id"] for r in rows("mls")}
        self.assertNotIn("m2", mls)   # "le cœur de la ville" == excluded "Le coeur de la ville" under norm fr
        self.assertIn("Bonjour à tous, merci. »", [r["text"] for r in rows("tedx")])   # natural ending in fr
        # vp: only OUR speakers excluded (v3); v1 / None kept although in VP train
        self.assertEqual({r["utt_id"] for r in rows("vp")}, {"vp0", "vp2"})
        sm = rows("smassive")
        self.assertEqual({r["group"] for r in sm}, {"sa", "sb", "sc"})   # strain / s115 dropped, sb only validated
        self.assertEqual({r["utt_id"] for r in sm}, {"h00", "h04", "h10", "h14"})
        self.assertTrue(all(r["bank"] == "smassive" for r in sm))

    def test_plan_only(self):
        d, log, out, rows = self.build("fr", ["--plan-only"])
        self.assertFalse([e for e in log if e[0] == "urlopen" or (e[0] == "fetch" and not e[1].endswith(".tsv"))])
        self.assertFalse(list(Path(f"{d}/out").glob("final*")) or Path(f"{d}/out/SHA256SUMS").exists())
        self.assertFalse(Path(f"{d}/audio").exists())
        self.assertIn("== PLAN-ONLY FINAL-FR", out)
        self.assertIn("duration unchecked", out)
        plan = lambda b: [json.loads(x)["utt_id"] for x in Path(f"{d}/out/plan_{b}.json").read_text().splitlines()]  # noqa: E731,E501
        _, _, _, full = self.build("fr")
        for b in ("mls", "tedx"):   # sources with a duration column: plan == the real selection
            self.assertEqual(plan(b), [r["utt_id"] for r in full(b)], b)
        self.assertEqual(set(plan("smassive")), {"h00", "h04", "h10", "h14"})
        self.assertEqual(set(plan("fleurs")), {"dev0", "dev1", "dev2"})   # audio presence not checked in plan
        Path(f"{d}/out/SHA256SUMS").write_text("sealed\n")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.drive(self.mod, d, ["--lang", "fr", "--plan-only", "--exclude-texts", "none"], self.world("fr"))

    def test_bank_language_guard(self):
        for lang, bank in (("fr", "italic"), ("it", "smassive"), ("fr", "nope")):
            with tempfile.TemporaryDirectory() as d, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.drive(self.mod, d, ["--lang", lang, "--banks", bank, "--plan-only", "--exclude-texts", "x"],
                         self.world(lang))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    res = unittest.TextTestRunner(verbosity=1, stream=sys.stderr).run(suite)
    skipped = len(res.skipped)
    for t, why in res.skipped:
        print(f"SKIP {t.id().split('.')[-1]}: {why}")
    ok = res.wasSuccessful()
    print(f"finetune self-test: {'PASS' if ok else 'FAIL'} ({res.testsRun} tests, {skipped} skipped)")
    sys.exit(0 if ok else 1)
