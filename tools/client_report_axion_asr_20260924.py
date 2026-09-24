#!/usr/bin/env python3
"""Nemotron streaming ASR on Arm - the client-facing PDF of 24 September 2026.

Same visual language as the 23 September report (tools/client_report_axion_asr.py):
its palette, table, callout and chart components are imported, not copied, so
the two documents read alike. The content is the day that followed: the core
plateau explained and removed, and the new operating point qualified with the
same fourteen bounds.

Every number comes from the qualification artefacts recorded in
.work/plateau-campaign-2026-09.md (DECISION RECORD) and
.work/nvidia-asr-streaming-landscape-20260923.md (S13-5c); nothing here is
computed at render time.

    uv run --with reportlab python3 tools/client_report_axion_asr_20260924.py OUT.pdf
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client_report_axion_asr import (ACCENT, BAD, GOOD, INK, MARG, MUTE, RULE,  # noqa: E402
                                     BODY, H1, H2, H3, MONO, SMALL, SUB, Callout,
                                     LagChart, LevelChart, Rule, st, table)
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether,  # noqa: E402
                                PageTemplate, Paragraph, Spacer)

FOOT = "Streaming ASR on Arm - GCP Axion c4a - 24 September 2026"


def build(path):
    doc = BaseDocTemplate(path, pagesize=A4,
                          leftMargin=20 * mm, rightMargin=18 * mm,
                          topMargin=17 * mm, bottomMargin=17 * mm,
                          title="Nemotron streaming ASR on Arm - 144 streams qualified",
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
    LEAD = st("lead24", fontSize=11, leading=16)

    # ------------------------------------------------------------------ title
    A(Paragraph("Streaming speech recognition on Arm", H1))
    A(Paragraph("144 concurrent live streams on one 32-core Arm server &mdash; "
                "GCP Axion c4a &middot; 24 September 2026", SUB))
    A(Spacer(1, 4)); A(Rule(fw, 1.2, ACCENT)); A(Spacer(1, 8))
    A(Callout(fw, [
        ("144", ["concurrent live streams", "qualified over 2 x 30 minutes"]),
        ("0", ["streams lost in 46,738", "utterances, 138 hours of audio"]),
        ("243 ms", ["emission lag, 95th percentile", "against a 320 ms bound"]),
        ("135x", ["realtime aggregate", "audio seconds per second"]),
    ]))
    A(Spacer(1, 10))
    A(Paragraph(
        "Yesterday this machine was qualified for 80 simultaneous live streams and "
        "left eight of its thirty server cores idle. Today it is qualified for 144 on "
        "the same hardware, the same model and the same fourteen bounds, and the "
        "transcripts are byte-identical to yesterday's &mdash; not one word changed. "
        "The gain came from finding where those eight cores went.", LEAD))
    A(Spacer(1, 6))
    A(Paragraph("Where it was yesterday and where it is now", H3))
    A(table([
        ["", "23 Sep", "24 Sep", "Change"],
        ["Qualified concurrency", "80", "144 (128 with wide margins)", "+80%"],
        ["Aggregate throughput", "77x realtime", "135x realtime", "+76%"],
        ["Emission lag, p95", "118 ms", "243 ms", "at 1.8x the load, bound 320"],
        ["Server cores actually used", "21.6&ndash;22.0 of 30", "26.9&ndash;27.4 of 30", "+5.3"],
        ["Streams lost", "0", "0", "&mdash;"],
        ["Transcripts", "reference", "498 of 498 byte-identical", "WER unchanged"],
        ["Hardware, model, layout, bounds", "c4a-highcpu-32, 0.6B int8, 6x5", "same", "&mdash;"],
    ], [fw * 0.27, fw * 0.21, fw * 0.28, fw * 0.24]))
    A(Paragraph(
        "The change is two serving settings that move work from each worker's "
        "scheduler thread onto its thread pool. They are switched on explicitly for "
        "this qualification and are not yet the shipped default (section 11).", SMALL))

    # ------------------------------------------------------------------ 1
    A(Paragraph("1. What is being measured", H2))
    A(table([
        ["Model", "nvidia/nemotron-3.5-asr-streaming-0.6b, the public pack, unmodified"],
        ["Architecture", "FastConformer encoder, RNN-T decoder, 0.6B parameters, cache-aware streaming"],
        ["Quantisation", "int8 weights, the shipped default"],
        ["Preset", "lookahead 3: one 320 ms chunk per step ((lookahead+1) x 80 ms)"],
        ["Audio", "16 kHz mono, sent in 100 ms frames at real time; English (FLEURS)"],
        ["Host", "GCP Axion c4a-highcpu-32, 32 x Arm Neoverse-V2, 62 GiB; CPU only, BLAS=none"],
        ["Server", "mynah-asr, C11; 6 workers x 5 threads pinned to cpus 0-29; load generator on 30-31"],
        ["Settings under test", "MYNAH_ASR_STREAM_PAR=4, MYNAH_ASR_FIN_STACK=1; everything else as on 23 Sep"],
        ["Build", "commit dbae3fe, clean tree; model pack sha256 2f5e1434..."],
    ], [fw * 0.24, fw * 0.76], header=False))

    # ------------------------------------------------------------------ 2
    A(Paragraph("2. What &ldquo;qualified&rdquo; means &mdash; unchanged", H2))
    A(Paragraph(
        "The fourteen bounds are the ones registered before the first measured run on "
        "22 September and used for the 80-stream qualification. None was relaxed. A "
        "level is qualified only by two thirty-minute runs, each on a freshly started "
        "server with a different random schedule; short runs may disqualify a level "
        "and never promote one.", BODY))
    A(table([
        ["Bound", "Threshold", "Bound", "Threshold"],
        ["Established streams lost", "0", "Client-observable stall", "max lag &le; 3 s"],
        ["Emission lag, p95", "&le; 320 ms", "Transcript parity", "byte-identical to unloaded"],
        ["Finalization, p95", "&le; 500 ms", "Worker RSS growth", "&le; 1.15x"],
        ["Backlog, max", "&le; 0.64 s", "Worker deaths", "0"],
        ["Per-window drift", "every 60 s window in bound", "TTFP load penalty, p95", "&le; 250 ms, paired"],
        ["Trend", "last third &le; 1.5x first", "Ready to first partial, p95", "&le; one cadence"],
        ["Server-side stall", "0 stuck intervals", "Admission refusals", "counted, not a loss"],
    ], [fw * 0.25, fw * 0.23, fw * 0.27, fw * 0.25]))

    # ------------------------------------------------------------------ 3
    A(KeepTogether([Paragraph("3. Results", H2), LevelChart(fw, [
        ("C80", 30, "GOOD", "23 Sep, shipped"),
        ("C104", 3, "FAILED", "shipped, 3 bounds"),
        ("C128", 30, "GOOD", "24 Sep, QUALIFIED x2"),
        ("C144", 30, "GOOD", "24 Sep, QUALIFIED x2"),
        ("C160", 2, "FAILED", "screen, first bad"),
    ])]))
    A(Paragraph(
        "Bar height is run length, not performance. The shipped configuration broke "
        "between 96 and 104 streams; with the two settings on, 128 and 144 each held "
        "two thirty-minute runs and 160 failed a screen.", SMALL))
    A(Spacer(1, 6))
    A(table([
        ["Level", "Settings", "Run", "Utts", "Lag p95", "Backlog", "Final p95", "Lost", "Verdict"],
        ["C80", "shipped", "30 min x2", "21,287", "115/118 ms", "0.384 s", "224/231 ms", "0", "QUALIFIED (23 Sep)"],
        ["C104", "shipped", "3 min", "1,285", "305 ms", "0.744 s", "619 ms", "0", "FAILED"],
        ["C128", "new", "30 min", "17,007", "129 ms", "0.284 s", "221 ms", "0", "QUALIFIED"],
        ["C128", "new", "30 min", "17,005", "129 ms", "0.284 s", "221 ms", "0", "QUALIFIED"],
        ["C144", "new", "30 min", "23,371", "243 ms", "0.384 s", "428 ms", "0", "QUALIFIED"],
        ["C144", "new", "30 min", "23,367", "243 ms", "0.464 s", "429 ms", "0", "QUALIFIED"],
        ["C160", "new", "2 min", "1,196", "761 ms", "1.084 s", "1338 ms", "0", "FAILED"],
    ], [fw * 0.08, fw * 0.10, fw * 0.11, fw * 0.11, fw * 0.12, fw * 0.10, fw * 0.12, fw * 0.07, fw * 0.19]))
    A(Paragraph(
        "The Lost column is zero on every row again, including the failed ones. At 160 "
        "streams the server is late, not lossy.", SMALL))
    A(Spacer(1, 8))
    A(LagChart(fw, {
        "shipped": ([(64, 80), (80, 108), (88, 133), (96, 182), (104, 305), (112, 556), (120, 1328)], MARG),
        "new settings": ([(128, 129), (144, 243), (160, 761)], ACCENT),
    }))
    A(Paragraph(
        "Same machine, model, corpus and layout. The shipped curve is the 23 September "
        "screen; the new points at 128 and 144 are thirty-minute runs, 160 is a screen. "
        "Points above 620 ms are drawn at the top of the axis.", SMALL))

    # ------------------------------------------------------------------ 4
    A(Paragraph("4. The operating point in detail", H2))
    A(Paragraph("C144, two independent thirty-minute runs, fresh server each:", BODY))
    A(table([
        ["Utterances completed", "46,738 (23,371 + 23,367), none dropped, none timed out"],
        ["Audio transcribed", "497,876 seconds &mdash; 138.3 hours"],
        ["Established streams lost / refusals", "0 / 0"],
        ["Emission lag", "p50 165 ms, p95 243 ms, p99 343&ndash;344 ms; worst delta 582 ms"],
        ["Finalization", "p50 303 ms, p95 428&ndash;429 ms (bound 500), p99 476&ndash;478 ms"],
        ["Backlog, max", "0.384 s and 0.464 s (bound 0.64 s)"],
        ["Late deltas", "of ~531,000 published per run, ~1.4% crossed one cadence and none crossed two"],
        ["First-word load penalty, paired p95", "+205 and +206 ms (GOOD &le; 250)"],
        ["Transcript parity", "498 of 498 clips identical to the unloaded transcript, both runs"],
        ["Throughput", "135.2x and 135.6x realtime"],
        ["CPU actually used", "26.9 and 27.4 cores of 30 (23 Sep: 21.6 and 22.0)"],
    ], [fw * 0.32, fw * 0.68], header=False))
    A(Paragraph(
        "<b>C128, the wide-margin alternative:</b> 34,012 utterances, 0 lost, lag p95 129 ms, "
        "finalization p95 221 ms, backlog max 0.284 s, first-word penalty p95 +103 ms, "
        "123.7x realtime on 26.2 cores. At 144 two margins are thin: finalization runs at "
        "86% of its bound and the first-word penalty at 82% of GOOD.", BODY))

    # ------------------------------------------------------------------ 5
    A(Paragraph("5. Where the eight cores went", H2))
    A(Paragraph(
        "Each worker is one scheduler thread plus a pool of four more threads. The "
        "pool did the big matrix products; everything that happens <i>per stream</i> "
        "&mdash; attention over each stream's cache, its convolution, layer norms, the "
        "decoder, and above all the last chunk of a finishing stream &mdash; ran on the "
        "scheduler thread alone while the pool waited. Measured from yesterday's runs, "
        "that serial share was about a quarter of the wall clock at 96 streams and "
        "grew with load, because it scales with the number of streams in a step.", BODY))
    A(table([
        ["Hypothesis tested", "Result"],
        ["Cache copying (a K/V memmove, 4.5 GB/s)", "Removed 93% of it: +1.7% throughput. Not the cause"],
        ["Pool threads falling asleep between tasks", "Kept them spinning: +5.5 cores burned, no gain. Not the cause"],
        ["A batching wait", "The qualified layout has none. Not the cause"],
        ["Per-stream stages on the pool", "C112 turned from failing to passing; &minus;66% lag p95"],
        ["Finishing streams on the batched path", "60 &rarr; 17 ms per finish; C128 from failing to passing"],
    ], [fw * 0.45, fw * 0.55]))
    A(Paragraph(
        "Each change was proved to produce the same floating-point results as the "
        "code it replaces before it was measured, and each was measured against its "
        "predecessor in interleaved repeats, not all-baseline-then-all-treatment.", SMALL))

    # ------------------------------------------------------------------ 6
    A(Paragraph("6. What to run on this machine", H2))
    A(table([
        ["Qualified concurrency", "144 streams", "2 x 30 min, 46,738 utterances, 0 lost"],
        ["With wide margins", "128 streams", "Every bound at &le; 45% of its limit"],
        ["Settings", "STREAM_PAR=4, FIN_STACK=1", "6 workers x 5 threads, cpus 0-29"],
        ["Expected emission lag", "243 ms p95 at 144; 129 ms at 128", "Bound 320 ms"],
        ["Expected throughput", "135x realtime at 144", "123.7x at 128"],
        ["First failed level", "160 streams (screen)", "Lag, finalization and backlog together"],
    ], [fw * 0.26, fw * 0.32, fw * 0.42]))
    A(Paragraph(
        "<b>Sizing by demand.</b> At 144 streams per machine, a peak of 1,000 concurrent "
        "speakers needs 7 machines of this class (yesterday 13); 2,500 needs 18 "
        "(yesterday 32). At the wide-margin 128: 8 and 20.", BODY))

    # ------------------------------------------------------------------ 7
    A(Paragraph("7. How long before the first word appears", H2))
    A(table([
        ["What", "C128", "C144", "Who owns it"],
        ["Speech onset to first partial, p50 / p95", "897 / 1,996 ms", "972 / 1,943 ms", "mostly the checkpoint"],
        ["First-word load penalty, p95, paired", "+103 ms", "+205 ms", "the server"],
        ["Ready to first partial, p95", "+102 ms", "+205 ms", "the server (bound 320)"],
    ], [fw * 0.40, fw * 0.17, fw * 0.17, fw * 0.26]))
    A(Paragraph(
        "The server's share grows with load, as it should, and stays inside its bound "
        "at both levels. The model's share is unchanged by serving work. We also tested "
        "whether the decoder could commit to its first word earlier: it can publish the "
        "first <i>fragment</i> of a word earlier (median &minus;180 ms), but the first "
        "<i>complete</i> word does not move at any setting and accuracy drops, so that "
        "knob stays off. Speaker-perceived first-word latency remains about two seconds "
        "at the 95th percentile on this corpus, outside our 1.5 s guideline, and is a "
        "property of the checkpoint.", BODY))

    # ------------------------------------------------------------------ 8
    A(Paragraph("8. Recognition quality", H2))
    A(table([
        ["Serving correctness", "Every transcript the new settings produce is byte-identical to the shipped "
         "build's, on 498 of 498 clips, idle and under load at 128 and 144 streams."],
        ["Model quality", "WER 10.4% mean / 6.7% median on the 349 original FLEURS recordings "
         "(10.6% mean on all 498), CER 6.9% mean &mdash; the same numbers as yesterday, by identity. "
         "Reported, not gated."],
    ], [fw * 0.24, fw * 0.76], header=False))

    # ------------------------------------------------------------------ 9
    A(Paragraph("9. The audio in the accompanying archive", H2))
    A(Paragraph(
        "Ten clips, chosen by a rule fixed before any new output was read: low, median and "
        "high word error; median, 95th and 99th percentile first-word time; a median clip "
        "of each length class; and one two-sentence concatenation. Each folder holds "
        "<font face='Courier'>sample.wav</font>, <font face='Courier'>reference.txt</font>, "
        "<font face='Courier'>baseline_asr.txt</font> (shipped settings), "
        "<font face='Courier'>candidate_asr.txt</font> (new settings) and "
        "<font face='Courier'>timing.json</font> (every partial with its time, for the "
        "first-word experiment); <font face='Courier'>index.csv</font> maps them all. "
        "Transcripts come from the server with the machine idle; under load they are "
        "identical by the parity bound.", BODY))

    # ------------------------------------------------------------------ 10
    A(Paragraph("10. Scope of these numbers", H2))
    A(table([
        ["English only", "The pack serves 40 languages. No validated French evaluation set exists in this "
         "campaign yet; French is unmeasured, not assumed."],
        ["One host class", "32-core Arm Neoverse-V2. Other hosts need their own screen."],
        ["Generator on the box", "The load client used 2 of the 32 cores; conservative."],
        ["Thin margins at 144", "Finalization at 86% of its bound. 128 is the choice for headroom."],
        ["Not yet the default", "The two settings are qualified here but not switched on in the shipped build."],
    ], [fw * 0.24, fw * 0.76], header=False))

    # ------------------------------------------------------------------ 11
    A(Paragraph("11. What comes next", H2))
    A(table([
        ["Make it the default", "Switch the two settings on in the shipped profile, choosing 144 or 128."],
        ["Simplify", "Remove the research variants that did not earn their place."],
        ["French", "Build a validated FR set and put it through the same quality gate."],
        ["x86, second generator host", "The same protocol on AMD/Intel; the client off the server."],
    ], [fw * 0.28, fw * 0.72], header=False))

    # ------------------------------------------------------------------ 12
    A(Paragraph("12. Reproducing this", H2))
    A(Paragraph(
        "MYNAH_ASR_STREAM_PAR=4 MYNAH_ASR_FIN_STACK=1 \\<br/>"
        "tools/bench/v2_qualify.sh -m models/nemotron-3.5-asr-streaming-0.6b \\<br/>"
        "&nbsp;&nbsp;--phase all --ladder \"\" --ref-c 4 --soak-c 144 --soak-seconds 1800 --soaks 2 \\<br/>"
        "&nbsp;&nbsp;--corpus samples/stress-en/manifest.json --corpus-sample 500 --corpus-seed 42 \\<br/>"
        "&nbsp;&nbsp;-W 6 -T 5 -C 96 --http-threads 32 --server-cpus 0-29 --gen-cpus 30-31<br/>"
        "python3 tools/bench/v2_verdict.py &lt;run&gt;", MONO))
    A(Spacer(1, 6))
    A(Paragraph(
        "Measurements taken 24 September 2026 on GCP Axion c4a-highcpu-32. Model "
        "nvidia/nemotron-3.5-asr-streaming-0.6b, int8, preset [56, 3]. Corpus FLEURS "
        "English (CC-BY 4.0), bank 04a7753aa1e80f9a, 498 clips. Runtime mynah-asr, C11. "
        "Commit dbae3fe. Prepared by Gabriele Mastrapasqua.", SMALL))
    doc.build(F)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "asr-report-20260924.pdf")
