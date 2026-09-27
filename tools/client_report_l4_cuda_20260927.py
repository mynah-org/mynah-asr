#!/usr/bin/env python3
"""Nemotron streaming ASR on one NVIDIA L4 - the client-facing PDF of the CUDA
engine (S14), 26-27 September 2026.

Same visual language as the Axion reports: the components of
tools/client_report_axion_asr.py are imported, not copied. Every number comes
from the artefacts summarised in .work/cuda-batched-streaming-server.md (raw
evidence untracked under .work/evidence/gpu-l4-20260926/); nothing is computed
at render time.

    uv run --with reportlab python3 tools/client_report_l4_cuda_20260927.py OUT.pdf
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client_report_axion_asr import (ACCENT, BAD, GOOD, INK, MARG, MUTE, RULE,  # noqa: E402
                                     BODY, H1, H2, H3, MONO, SMALL, SUB, Callout,
                                     LagChart, LevelChart, Rule, st, table)
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, KeepTogether,  # noqa: E402
                                PageTemplate, Paragraph, Spacer)

FOOT = "Streaming ASR on one NVIDIA L4 - the CUDA engine - 27 September 2026"


class StageChart(Flowable):
    """Device milliseconds per pass, stacked by stage, one bar per arm."""
    STAGES = [("FFN (2 GEMMs x 2)", colors.HexColor("#1f4e79")),
              ("attention projections + O", colors.HexColor("#4f7fae")),
              ("attention core", colors.HexColor("#b07d1c")),
              ("conv module", colors.HexColor("#7d9c4f")),
              ("decoder (joint, predictor, sync)", colors.HexColor("#8a5a9e")),
              ("subsampling, post, transfers", colors.HexColor("#9a9a9a"))]

    def __init__(self, width, bars):
        self.width, self.bars = width, bars
        self.height = 16 * mm + 11 * mm * len(bars)

    def draw(self):
        c = self.canv
        L, R = 150, 38
        pw = self.width - L - R
        hi = 80.0
        c.setFont("Helvetica-Bold", 8.5); c.setFillColor(INK)
        c.drawString(0, self.height - 8, "Where one pass spends the GPU: device ms per pass, by stage")
        top = self.height - 22
        for i, (label, sub, vals) in enumerate(self.bars):
            y = top - (i + 1) * 11 * mm + 4 * mm
            c.setFillColor(INK); c.setFont("Helvetica-Bold", 7.8)
            c.drawString(0, y + 5, label)
            c.setFillColor(MUTE); c.setFont("Helvetica", 6.6)
            c.drawString(0, y - 3, sub)
            x = L
            for (name, col), v in zip(self.STAGES, vals):
                w = pw * v / hi
                c.setFillColor(col); c.rect(x, y - 4, w, 14, stroke=0, fill=1)
                if w > 18:
                    c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 6.4)
                    c.drawCentredString(x + w / 2, y + 1, "%.1f" % v)
                x += w
            c.setFillColor(INK); c.setFont("Helvetica-Bold", 7.6)
            c.drawString(x + 4, y + 1, "%.1f ms" % sum(vals))
        # legend
        lx, ly = 0, 6
        c.setFont("Helvetica", 6.6)
        for name, col in self.STAGES:
            c.setFillColor(col); c.rect(lx, ly, 7, 7, stroke=0, fill=1)
            c.setFillColor(MUTE); c.drawString(lx + 10, ly + 1, name)
            lx += 10 + c.stringWidth(name, "Helvetica", 6.6) + 12


class GemmChart(Flowable):
    """One GEMM (the FFN down-projection, N=1024 K=4096) against cohort size."""
    def __init__(self, width, rows):
        self.width, self.rows = width, rows
        self.height = 58 * mm

    def draw(self):
        c = self.canv
        L, R, B = 34, 10, 24
        pw = self.width - L - R
        ph = self.height - B - 30
        hi = 0.5
        arms = [("v1 (row-stable, one chain)", colors.HexColor("#9a9a9a")),
                ("split-K (row-stable by construction)", ACCENT),
                ("cuBLAS (NOT row-stable)", MARG)]
        c.setFont("Helvetica-Bold", 8.5); c.setFillColor(INK)
        c.drawString(0, self.height - 8, "FFN down-projection (N=1024, K=4096): milliseconds per call by cohort rows")
        c.setFont("Helvetica", 7); c.setFillColor(MUTE)
        for v in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            y = B + ph * v / hi
            c.setStrokeColor(RULE); c.setLineWidth(0.3); c.line(L, y, L + pw, y)
            c.drawRightString(L - 4, y - 2.5, "%.1f" % v)
        slot = pw / len(self.rows)
        bw = slot * 0.24
        for i, (m, vals) in enumerate(self.rows):
            x0 = L + i * slot + slot * 0.14
            for j, ((name, col), v) in enumerate(zip(arms, vals)):
                h = ph * v / hi
                c.setFillColor(col); c.rect(x0 + j * bw, B, bw * 0.92, h, stroke=0, fill=1)
            c.setFillColor(INK); c.setFont("Helvetica-Bold", 7.6)
            c.drawCentredString(x0 + 1.5 * bw, B - 10, "M=%d" % m)
        c.setStrokeColor(MUTE); c.setLineWidth(0.6); c.line(L, B, L + pw, B)
        lx = L
        c.setFont("Helvetica", 6.8)
        for name, col in arms:
            c.setFillColor(col); c.rect(lx, self.height - 22, 7, 7, stroke=0, fill=1)
            c.setFillColor(MUTE); c.drawString(lx + 10, self.height - 21, name)
            lx += 10 + c.stringWidth(name, "Helvetica", 6.8) + 14


class LagChart2(Flowable):
    """LagChart of the Axion reports with a per-series label offset (two series
    that end at the same point would print their names on top of each other)
    and the bound's label on the left, clear of the points."""
    def __init__(self, width, series, bound=320.0):
        self.width, self.series, self.bound = width, series, bound
        self.height = 54 * mm

    def draw(self):
        c = self.canv
        L, R, B = 32, 40, 26
        pw = self.width - L - R
        ph = self.height - B - 18
        lo, hi = 0.0, 650.0
        ypix = lambda v: B + ph * (v - lo) / (hi - lo)
        xs = [p[0] for pts, _, _ in self.series.values() for p in pts]
        xlo, xhi = min(xs) - 4, max(xs) + 4
        xpix = lambda v: L + pw * (v - xlo) / (xhi - xlo)
        c.setFont("Helvetica-Bold", 8.5); c.setFillColor(INK)
        c.drawString(0, self.height - 8, "Emission lag, 95th percentile, against the 320 ms cadence (90 s screens)")
        c.setFont("Helvetica", 7); c.setFillColor(MUTE)
        for v in (0, 100, 200, 300, 400, 500, 600):
            y = ypix(v)
            c.setStrokeColor(RULE); c.setLineWidth(0.3); c.line(L, y, L + pw, y)
            c.drawRightString(L - 4, y - 2.5, "%d" % v)
        c.saveState(); c.translate(9, B + ph / 2); c.rotate(90)
        c.setFont("Helvetica", 7.2); c.drawCentredString(0, 0, "lag p95 (ms)")
        c.restoreState()
        yb = ypix(self.bound)
        c.setStrokeColor(BAD); c.setLineWidth(1); c.setDash(3, 2)
        c.line(L, yb, L + pw, yb); c.setDash()
        c.setFillColor(BAD); c.setFont("Helvetica-Bold", 6.8)
        c.drawString(L + 4, yb + 4, "320 ms bound = (lookahead+1) x 80")
        for name, (pts, col, dy) in self.series.items():
            pts = sorted(pts, key=lambda p: p[0])
            for i in range(1, len(pts)):
                c.setStrokeColor(col); c.setLineWidth(1.0)
                c.line(xpix(pts[i - 1][0]), ypix(pts[i - 1][1]), xpix(pts[i][0]), ypix(pts[i][1]))
            for cc, val in pts:
                x, y = xpix(cc), ypix(val)
                c.setFillColor(colors.white); c.circle(x, y, 3.8, stroke=0, fill=1)
                c.setFillColor(col); c.circle(x, y, 2.7, stroke=0, fill=1)
                c.setFillColor(col); c.setFont("Helvetica", 6.4)
                c.drawCentredString(x, y + (6 if dy >= 0 else -11), "%d" % val)
            lx, ly = xpix(pts[-1][0]), ypix(pts[-1][1])
            c.setFillColor(col); c.setFont("Helvetica-Bold", 7)
            c.drawString(lx + 8, ly - 2 + dy, name)
        c.setFillColor(INK); c.setFont("Helvetica-Bold", 7.6)
        for cc in sorted({p[0] for pts, _, _ in self.series.values() for p in pts}):
            c.drawCentredString(xpix(cc), B - 10, "C%d" % cc)
        c.setStrokeColor(MUTE); c.setLineWidth(0.6); c.line(L, B, L + pw, B)


def build(path):
    doc = BaseDocTemplate(path, pagesize=A4,
                          leftMargin=20 * mm, rightMargin=18 * mm,
                          topMargin=17 * mm, bottomMargin=17 * mm,
                          title="Nemotron streaming ASR on one NVIDIA L4 - the CUDA engine",
                          author="Gabriele Mastrapasqua")
    fw = doc.width
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="n")

    def decorate(canv, d):
        canv.saveState()
        canv.setFont("Helvetica", 7.2); canv.setFillColor(MUTE)
        canv.drawString(doc.leftMargin, 11 * mm, FOOT)
        canv.drawRightString(doc.leftMargin + doc.width, 11 * mm, "page %d" % d.page)
        canv.setStrokeColor(RULE); canv.setLineWidth(0.4)
        canv.line(doc.leftMargin, 13.5 * mm, doc.leftMargin + doc.width, 13.5 * mm)
        canv.restoreState()

    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=decorate)])
    F = []
    A = F.append
    LEAD = st("lead27", fontSize=11, leading=16)
    WIN = '<font color="#2f7d4f"><b>WIN</b></font>'
    FAIL = '<font color="#a33a2e"><b>FAIL</b></font>'
    TODO = '<font color="#b07d1c"><b>TODO</b></font>'
    OPEN = '<font color="#b07d1c"><b>OPEN</b></font>'

    # ------------------------------------------------------------------ title
    A(Paragraph("Streaming speech recognition on one NVIDIA L4", H1))
    A(Paragraph("Nemotron 0.6B on the mynah-asr CUDA engine &mdash; design, bring-up, first "
                "qualification and the first optimisation &middot; 26-27 September 2026", SUB))
    A(Spacer(1, 4)); A(Rule(fw, 1.2, ACCENT)); A(Spacer(1, 8))
    A(Callout(fw, [
        ("0", ["streams lost in 78,738 utterances", "across both qualifications"]),
        ("498 / 498", ["bank transcripts identical", "GPU split-K vs GPU v1"]),
        ("-31 %", ["GPU time per step after", "the first optimisation"]),
        ("C128 / C144", ["30-min qualification missed", "by v1 / by split-K"]),
    ]))
    A(Spacer(1, 10))
    A(Paragraph(
        "The same Nemotron streaming model that the CPU server qualifies at 144 simultaneous "
        "streams on a 32-core Arm machine now runs on a single NVIDIA L4 (72 W, 24 GB), in a "
        "separate CUDA engine written for the purpose: the weights and every stream's state live "
        "on the GPU and all ready streams are advanced together in one batched step. It speaks "
        "the same protocol, is judged by the same harness, the same corpus and the same "
        "fourteen bounds, and gives the same words: on the GPU the transcripts equal the CPU's. "
        "In one day it went from nothing to 128 streams inside every latency bound on a "
        "90-second screen; the thirty-minute qualification, which alone certifies a level, was "
        "missed by 44 ms of queue at 128 streams, and after the first optimisation by a wider "
        "margin at 144. No level is certified on the GPU yet; this report says exactly where "
        "each one stands.", LEAD))
    A(Spacer(1, 6))
    A(Paragraph("Where it started and where it is now", H3))
    A(table([
        ["", "Before (25 Sep)", "CUDA F32 v1 (26 Sep)", "+ split-K GEMM (26 Sep)"],
        ["GPU serving path", "none (a July CLI offload of single GEMMs)", "resident engine, batched step", "same"],
        ["GPU time per step, C128", "&mdash;", "54.5 ms", "37.6 ms (-31 %)"],
        ["90 s screen at C128: lag / finalization p95", "&mdash;", "165 / 224 ms", "97 / 128 ms"],
        ["90 s screen at C144", "&mdash;", "fails (backlog 0.78 s)", "passes (166 / 216 ms)"],
        ["30 min x 2 at C128", "&mdash;", "1 pass, 1 miss by 44 ms", "not run yet"],
        ["30 min x 2 at C144", "&mdash;", "not run", "fails (lag 346 ms, backlog 1.06 s)"],
        ["Transcripts under load", "&mdash;", "identical, 498 clips", "identical to v1, 498 clips"],
    ], [fw * 0.31, fw * 0.21, fw * 0.24, fw * 0.24]))
    A(Paragraph("For orientation, not as a claim: the CPU fleet on the 32-core Axion is "
                "certified at 144 English streams, and on the same bank at 128 streams it reads lag "
                "p95 129 ms, finalization 221 ms and about 124 seconds of speech per second. The "
                "two machines differ in class and cost; the comparison that matters comes once the "
                "GPU has a certified level.", SMALL))

    # ------------------------------------------------------------------ 1
    A(Paragraph("1. What is being measured", H2))
    A(table([
        ["Model", "nvidia/nemotron-3.5-asr-streaming-0.6b, the public pack, unmodified; lookahead 3 (one 320 ms chunk per step)"],
        ["Precision", "f32 weights resident on the GPU (2.35 GB), f32 arithmetic, no TF32; 73 MB of per-stream state for 192 slots"],
        ["GPU", "NVIDIA L4, sm_89, 22.5 GB usable (ECC on), 72 W; rented container (vast.ai), 24.5 CPU quota on a shared host"],
        ["Engine", "mynah-asr CUDA engine, own kernels (C11 + CUDA); one process, one GPU, one engine thread"],
        ["Batching", "a cohort of every stream with a ready chunk, gathered for up to 40 ms, stepped together; "
         "first chunks, steady chunks and final tails mixed in one pass"],
        ["Corpus", "FLEURS English stress bank, the 498-clip qualification sample (hash 04a7753aa1e80f9a), the "
         "same one the CPU qualifications used"],
        ["Bounds", "the fourteen registered on 22 September plus the accounting rows of 25 September, judged by "
         "the unmodified verdict tool; none relaxed"],
    ], [fw * 0.17, fw * 0.83], header=False))

    # ------------------------------------------------------------------ 2
    A(Paragraph("2. The design in one paragraph", H2))
    A(Paragraph(
        "The CPU server was not ported. A new tree (<font face='Courier'>gpu/</font>) holds a "
        "separate server binary and engine; the qualified CPU path is untouched and serves as "
        "the library for loading the model, the tokenizer and the audio front end. Everything the "
        "GPU computes stays on the GPU between steps: the 24-layer encoder's attention cache, its "
        "convolution cache, the subsampling caches and the decoder state of every stream. A step "
        "uploads only the new audio features and a small table describing each stream, and "
        "downloads only the new tokens. The decoder runs on the GPU too, all streams together, "
        "one emitted token per loop. The protocol, the refusals, the session accounting and the "
        "metrics are the CPU server's, so every client, probe and qualification tool works "
        "against it unchanged.", BODY))
    A(Paragraph(
        "One rule shapes the arithmetic: <b>a stream's words must never depend on which other "
        "streams it was batched with</b>. On the GPU this is not free &mdash; the vendor matrix "
        "library chooses a different internal algorithm for each batch size &mdash; so the "
        "engine uses its own matrix kernels, built so that each output row depends only on its "
        "own input row, and the tests check it byte by byte at every batch size.", BODY))

    # ------------------------------------------------------------------ 3
    A(Paragraph("3. Is the GPU what it claims to be?", H2))
    A(Paragraph("Checked before any number was taken, because it is a rented container:", BODY))
    A(table([
        ["Check", "Measured", "Reading"],
        ["Power state under load", "P0, memory at 6,251 MHz", "full clocks"],
        ["Power limit", "72 W = default = maximum", "not lowered by the provider"],
        ["Sharing", "no MIG partition, no other process", "the whole GPU"],
        ["Memory allocatable", "22,272 of 22,565 MB (98.7 %)", "no cap"],
        ["Memory bandwidth", "210 GB/s (70 % of 300)", "normal for GDDR6"],
        ["FP32 sustained", "23.8 TFLOP/s (78 % of 30.3)", "the L4's own 72 W limit holds the clock at ~1,570 of 2,040 MHz; the same on any L4"],
        ["PCIe", "25.4 / 22.1 GB/s", "Gen4 x16"],
    ], [fw * 0.24, fw * 0.32, fw * 0.44]))

    # ------------------------------------------------------------------ 4
    A(Paragraph("4. Correctness before speed", H2))
    A(table([
        ["Gate", "Result"],
        ["Every GPU kernel against a CPU reference", WIN + " 51 of 51, first run on the device"],
        ["Matrix kernel: same row, same bits, at every batch size", WIN + " all 12 shapes of the model, 1 to 257 rows"],
        ["Batch identity: a stream alone = the same stream in a mixed batch, on any slot", WIN + " 5 clips, 5 languages"],
        ["GPU transcripts = CPU transcripts (f32)", WIN + " identical on 5 of 5 clips, 5 languages"],
        ["Protocol probes of the CPU server, unchanged", WIN + " 7 of 7 (after two fixes below)"],
        ["Fault suite: resets, half-closes, stalls, garbage, capacity, 40 aborts", WIN + " 16 cases, 0 failed invariants, 0.00 s of work for departed clients"],
        ["Session accounting under load", WIN + " balanced in every one of 250+ snapshots"],
    ], [fw * 0.56, fw * 0.44]))
    A(Paragraph(
        "The first real-GPU run found two defects that the engine-level tests could not: the "
        "server told the engine to finish an utterance while seconds of its audio were still "
        "queued (empty transcripts), and repeated resets of an idle stream overflowed a small "
        "transfer buffer (the engine stopped, visibly, as designed). Both were fixed the same "
        "hour and the whole suite re-run green.", BODY))

    # ------------------------------------------------------------------ 5
    A(Paragraph("5. The first qualification: CUDA F32 v1", H2))
    A(LevelChart(fw, [
        ("C96", 1.5, "GOOD", "screen"),
        ("C128", 1.5, "GOOD", "screen"),
        ("C144", 1.5, "FAILED", "screen"),
        ("C160", 1.5, "FAILED", "screen"),
        ("C128", 30, "FAILED", "soak 1: +44 ms"),
        ("C128", 30, "GOOD", "soak 2"),
    ]))
    A(table([
        ["Run", "Lost", "Lag p95", "Final. p95", "Backlog max", "Worst window", "Speech/s", "Verdict"],
        ["C96, 90 s", "0", "114 ms", "158 ms", "0.18 s", "116 ms", "63x", "pass"],
        ["C128, 90 s", "0", "165 ms", "224 ms", "0.34 s", "178 ms", "89x", "pass"],
        ["C144, 90 s", "0", "286 ms", "369 ms", "<b>0.78 s</b>", "<b>352 ms</b>", "103x", "fail"],
        ["C160, 90 s", "0", "<b>611 ms</b>", "<b>814 ms</b>", "<b>0.88 s</b>", "<b>675 ms</b>", "110x", "fail"],
        ["C128, 30 min, seed 42", "0 / 17,016", "177 ms", "237 ms", "<b>0.684 s</b>", "213 ms", "123x", "<b>fail by 44 ms</b>"],
        ["C128, 30 min, seed 43", "0 / 17,023", "164 ms", "223 ms", "0.584 s", "180 ms", "124x", "pass"],
    ], [fw * 0.20, fw * 0.10, fw * 0.10, fw * 0.10, fw * 0.12, fw * 0.12, fw * 0.10, fw * 0.16]))
    A(Paragraph(
        "Bounds: lag 320 ms (one chunk period), finalization 500 ms, backlog 0.64 s (two chunks). "
        "<b>C128 is not certified</b>: certification needs two passing thirty-minute runs and the "
        "first missed the backlog bound by 44 ms, with every other gate green. The server's own "
        "counters explained the edge before any profiler did: the engine thread was inside a "
        "step 86 % of the time, 68-71 ms per step, so a new batch waited for the previous one "
        "rather than for the 40 ms window.", BODY))

    # ------------------------------------------------------------------ 6
    sec6 = [Paragraph("6. Where the time goes", H2)]
    sec6.append(Paragraph(
        "A per-stage timer inside the engine (CUDA events, a diagnostic switch, off in "
        "production) measured C128 against C160 &mdash; one level that holds, one that does not. "
        "A table of what each outcome would mean was written down before the data came in.", BODY))
    sec6.append(StageChart(fw, [
        ("v1, C128", "95 rows per pass", [26.1, 8.6, 6.7, 6.0, 5.4, 1.7]),
        ("v1, C160", "163 rows per pass", [34.9, 11.0, 10.8, 7.9, 6.1, 2.3]),
        ("split-K, C128", "73 rows per pass", [15.7, 5.3, 6.1, 4.4, 4.7, 1.4]),
        ("split-K, C160", "125 rows per pass", [22.7, 7.4, 9.6, 6.1, 5.2, 1.8]),
    ]))
    A(KeepTogether(sec6))
    A(Spacer(1, 4))
    A(Paragraph(
        "Reading, from the registered table: the step time equals the GPU time (55.0 against "
        "54.5 ms), the audio front end on the host uses about 11 % of one thread, and the "
        "decoder's round trips to the host are 0.4 %. So the server, the host and the decoder "
        "loop are not the limit. <b>The matrix multiplications are about 75 % of the GPU time</b>, "
        "and the engine's own f32 kernel ran at about 27 % of what the L4 sustains.", BODY))

    # ------------------------------------------------------------------ 7
    A(Paragraph("7. The first optimisation: a deterministic split-K", H2))
    A(Paragraph(
        "The vendor library (cuBLAS) halved the step in a diagnostic run, but measured on this "
        "card it breaks the rule of section 2 on every one of the model's twelve matrix shapes: "
        "the same row gives different bits at different batch sizes. It cannot be the default. "
        "A faster version of the engine's own kernel that kept exactly the old arithmetic was "
        "built and measured: identical bits, but not faster &mdash; one sequential sum per output "
        "cannot fill the GPU when few streams share a step and the inner dimension is long. "
        "The lever that works changes the order of the sum in a controlled way: each sum is cut "
        "into pieces whose number depends only on the matrix's shape, never on how many streams "
        "are in the step, and the pieces are added back in a fixed order.", BODY))
    A(GemmChart(fw, [(24, [0.322, 0.100, 0.035]), (64, [0.310, 0.097, 0.053]),
                     (95, [0.310, 0.141, 0.091]), (128, [0.296, 0.145, 0.096]),
                     (163, [0.307, 0.208, 0.154]), (256, [0.487, 0.297, 0.181])]))
    A(table([
        ["What was checked", "Split-K result"],
        ["Same row, same bits at every batch size (12 shapes, 12 sizes)", WIN + " yes, by construction and by test"],
        ["Numerical error against a double-precision reference", WIN + " smaller than the old kernel on every shape (FFN: 1.3e-6 vs 5.4e-6)"],
        ["Speed per matrix call vs the old kernel", WIN + " 1.1x to 3.5x faster; still 1.3-1.7x behind cuBLAS"],
        ["Transcripts of the 498-clip bank vs the old kernel", WIN + " 498 of 498 identical; word error rate unchanged (mean 0.105)"],
        ["GPU time per step at C128", WIN + " 54.5 &rarr; 37.6 ms (-31 %); the FFN share 26.1 &rarr; 15.7 ms"],
    ], [fw * 0.52, fw * 0.48]))

    A(KeepTogether([Paragraph("The split-K qualification", H3), LevelChart(fw, [
        ("C128", 1.5, "GOOD", "screen"),
        ("C144", 1.5, "GOOD", "screen"),
        ("C152", 1.5, "FAILED", "screen"),
        ("C160", 1.5, "FAILED", "screen"),
        ("C144", 30, "FAILED", "soak 1"),
        ("C144", 30, "FAILED", "soak 2, not paced"),
    ])]))
    A(table([
        ["Run", "Lost", "Lag p95", "Final. p95", "Backlog max", "Worst window", "Trend", "Speech per s", "Verdict"],
        ["C128, 90 s", "0", "97 ms", "128 ms", "0.18 s", "97 ms", "&mdash;", "91x", "pass"],
        ["C144, 90 s", "0", "166 ms", "216 ms", "0.58 s", "258 ms", "&mdash;", "104x", "pass"],
        ["C152, 90 s", "0", "<b>360 ms</b>", "447 ms", "<b>0.68 s</b>", "<b>506 ms</b>", "&mdash;", "91x", "fail"],
        ["C160, 90 s", "0", "<b>607 ms</b>", "<b>781 ms</b>", "<b>0.98 s</b>", "<b>773 ms</b>", "&mdash;", "111x", "fail"],
        ["C144, 30 min, s42", "0 / 19,088", "<b>346 ms</b>", "455 ms", "<b>1.06 s</b>", "<b>515 ms</b>", "+11%", "138x", "fail"],
        ["C144, 30 min, s43", "0 / 19,022", "<b>467 ms</b>", "<b>625 ms</b>", "<b>1.16 s</b>", "<b>783 ms</b>", "+18%", "138x", "fail, invalid"],
    ], [fw * 0.18, fw * 0.10, fw * 0.09, fw * 0.09, fw * 0.10, fw * 0.10, fw * 0.08, fw * 0.10, fw * 0.16]))
    A(Paragraph(
        "The ninety-second screen moved the clean level from 128 to 144, but at 144 the queue "
        "grows over thirty minutes (+11 % trend): 144 is at the edge under sustained load. The "
        "second soak is also invalid as a measurement: the load generator shares a 24.5-CPU "
        "container on a busy multi-tenant host with the server and could not keep real time at "
        "144 clients. The split-K pair at 128, where the screen shows wide margins, was started "
        "and stopped at the end of the day before it measured anything.", BODY))
    A(KeepTogether([LagChart2(fw, {
        "v1": ([(96, 114), (128, 165), (144, 286), (160, 611)], colors.HexColor("#9a9a9a"), 8),
        "split-K": ([(128, 97), (144, 166), (152, 360), (160, 607)], ACCENT, -10),
    }), Paragraph("90-second screens on the qualification bank. Only thirty-minute pairs certify.", SMALL)]))

    # ------------------------------------------------------------------ 8
    A(Paragraph("8. Wins, fails and what is still open", H2))
    A(table([
        ["", "Item"],
        [WIN, "A separate CUDA engine and server, protocol- and harness-compatible, the qualified CPU path untouched"],
        [WIN, "GPU transcripts identical to the CPU's; batch identity proven at every batch size"],
        [WIN, "The full fault suite green on the GPU; exact session accounting; no work for departed clients"],
        [WIN, "The rented L4 verified: full clocks, full power limit, 98.7 % of memory, no other tenant on the GPU"],
        [WIN, "Profile with a pre-registered reading: the matrix kernels are the limit, not serving, host or decoder"],
        [WIN, "Split-K: -31 % GPU time per step, more accurate, bank transcripts unchanged"],
        [FAIL, "CUDA F32 v1, C128 thirty-minute pair: one run missed the backlog bound by 44 ms"],
        [FAIL, "Split-K, C144 thirty-minute pair: both runs over the lag and backlog bounds"],
        [FAIL, "cuBLAS as the matrix kernel: not row-stable on this card (kept only as a comparison)"],
        [FAIL, "A faster kernel keeping the old arithmetic exactly: identical bits, no speed-up (kept as a comparison)"],
        [OPEN, "Split-K at C128 over thirty minutes, with the load generator on another machine"],
        [TODO, "Decide whether split-K becomes the default (the quality evidence is already in)"],
        [TODO, "bf16 tensor cores, starting with the FFN (about half the GPU time), each step behind a word-error-rate gate"],
        [TODO, "The attention kernel, now the second largest stage (16-18 %)"],
        [TODO, "The target instance class: AWS g6.xlarge (L4, 4 vCPU); the account's G-instance vCPU quota must first be raised from 0"],
        [TODO, "Audio front end on the GPU, needed only if 4 vCPUs cannot carry it at the certified level"],
    ], [fw * 0.10, fw * 0.90]))

    # ------------------------------------------------------------------ 9
    A(Paragraph("9. Scope of these numbers", H2))
    A(table([
        ["One GPU, one host", "NVIDIA L4 in a rented container; not yet the AWS instance class it is aimed at"],
        ["Generator on the same host", "The load generator shared the container's CPUs and, at 144 clients, lost real time once; "
         "the next soaks move it off the host"],
        ["English only", "The FLEURS English bank; the CPU campaign's French validation has not been repeated on the GPU"],
        ["f32 only", "No reduced precision yet; bf16 is the next lever and carries its own quality gate"],
        ["No certified level", "Every GPU level in this report is either a screen or a failed pair"],
    ], [fw * 0.26, fw * 0.74], header=False))

    # ------------------------------------------------------------------ 10
    A(Paragraph("10. The work, in order", H2))
    A(table([
        ["When", "What"],
        ["26 Sep, morning", "Design and cost model; the engine, kernels, server, tests and a CUDA build in CI"],
        ["26 Sep, noon", "First GPU run: GPU checked, kernels and transcript gates green, two defects fixed, fault suite green"],
        ["26 Sep, afternoon", "V2 qualification of CUDA F32 v1; per-stage profile; cuBLAS measured and rejected; faster-exact kernel rejected"],
        ["26 Sep, evening", "Split-K built, gated and measured; bank quality unchanged; V2 qualification at C144 failed"],
        ["27 Sep", "This report; CI green on every job for the branch head"],
    ], [fw * 0.20, fw * 0.80]))
    A(Paragraph("18 commits on research/asr-streaming-lab (930c77f .. f0943d7), every one pushed; "
                "the continuous integration builds the CUDA tree for sm_80, sm_86, sm_89 and sm_90 "
                "on every push.", SMALL))

    # ------------------------------------------------------------------ 11
    A(Paragraph("11. Reproducing this", H2))
    A(Paragraph("The engine, the server, the tests and the qualification runner are open source "
                "and committed; the runner refuses a dirty tree, a GPU in use by anyone else and a "
                "busy container.", BODY))
    A(Paragraph(
        "make lib &amp;&amp; make -C gpu CUDA_ARCH=sm_89<br/>"
        "gpu/tools/gpu_doctor.sh --peak-bw-gbs 300 --peak-tflops 30.3<br/>"
        "tests/test_cuda_kernels &amp;&amp; tests/test_cuda_stream models/nemotron-3.5-asr-streaming-0.6b --gemm splitk<br/>"
        "gpu/tools/gpu_qualify.sh -m models/nemotron-3.5-asr-streaming-0.6b --gemm splitk \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--corpus samples/stress-en/manifest.json --corpus-sample 498 --soak-c 128", MONO))
    A(Spacer(1, 8)); A(Rule(fw, 0.4))
    A(Paragraph("Measurements taken 26 September 2026 on an NVIDIA L4 (vast.ai). Model "
                "nvidia/nemotron-3.5-asr-streaming-0.6b; corpus FLEURS (CC-BY 4.0). Runtime mynah-asr, "
                "C11 and CUDA, open source. Prepared by Gabriele Mastrapasqua.", SMALL))

    doc.build(F)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "report.pdf")
