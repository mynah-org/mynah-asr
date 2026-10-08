#!/usr/bin/env python3
"""Nemotron streaming ASR on Arm - the client-facing PDF of 25 September 2026.

Same visual language as the 23 and 24 September reports: the components of
tools/client_report_axion_asr.py are imported, not copied. The content is the
day that followed the 144-stream qualification: failures made measurable and
provoked on purpose, a real zombie inference found and removed, service-wide
metrics, English and French validated, the capacity knee closed at C=152, and
the first-word latency attributed.

Every number comes from the artefacts recorded in
.work/failure-accounting-and-cancellation.md, .work/french-validation-20260925.md,
.work/nvidia-asr-streaming-landscape-20260923.md (S13-10) and
.work/plateau-campaign-2026-09.md (C=152); nothing here is computed at render time.

    uv run --with reportlab python3 tools/client_report_axion_asr_20260925.py OUT.pdf
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

FOOT = "Streaming ASR on Arm - GCP Axion c4a - 25 September 2026"


def build(path):
    doc = BaseDocTemplate(path, pagesize=A4,
                          leftMargin=20 * mm, rightMargin=18 * mm,
                          topMargin=17 * mm, bottomMargin=17 * mm,
                          title="Nemotron streaming ASR on Arm - English and French validated",
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
    LEAD = st("lead25", fontSize=11, leading=16)

    # ------------------------------------------------------------------ title
    A(Paragraph("Streaming speech recognition on Arm", H1))
    A(Paragraph("English and French validated, failures accounted for &mdash; "
                "GCP Axion c4a &middot; 25 September 2026", SUB))
    A(Spacer(1, 4)); A(Rule(fw, 1.2, ACCENT)); A(Spacer(1, 8))
    A(Callout(fw, [
        ("0.00 s", ["of audio computed after a", "client vanished (was 27.4 s)"]),
        ("1,878", ["clients aborted on purpose,", "every one accounted for"]),
        ("EN + FR", ["validated on the qualified", "settings, under load"]),
        ("C128", ["recommended conservative", "EN+FR operating point"]),
    ]))
    A(Spacer(1, 10))
    A(Paragraph(
        "On 24 September this machine was qualified for 144 simultaneous English streams "
        "with zero streams lost. Today asked the harder question: when things go wrong "
        "&mdash; a client disconnects, stalls, sends garbage, a worker dies &mdash; does the "
        "server notice, stop working for nobody, free everything and count it correctly? It "
        "did not always: one real waste was found and removed. Then French was validated on "
        "the same settings, the capacity curve was closed, and the first-word latency was "
        "attributed to where it actually comes from.", LEAD))
    A(Spacer(1, 6))
    A(Paragraph("Where it was on 24 September and where it is now", H3))
    A(table([
        ["", "24 Sep", "25 Sep"],
        ["Client vanishes mid-utterance (silence)", "up to 27.4 s of audio computed for nobody, "
         "counted nowhere", "0.00 s, counted as a disconnect"],
        ["Session accounting", "client-side totals only", "exact server books, client and server "
         "reconciled session by session"],
        ["Provoked failures", "none", "17 fault cases + 2 fault soaks, all QUALIFIED"],
        ["Metrics under --prefork", "router view only", "service-wide, summed over workers"],
        ["French", "not measured", "validated; C128 passes under load"],
        ["Capacity knee (English)", "144 good, 160 bad", "144 good, 152 bad, 160 bad"],
        ["First word, p95 2.3 s", "unexplained", "attributed; the runtime adds 0 ms"],
    ], [fw * 0.30, fw * 0.33, fw * 0.37]))

    # ------------------------------------------------------------------ 1
    A(Paragraph("1. What is being measured", H2))
    A(table([
        ["Model", "nvidia/nemotron-3.5-asr-streaming-0.6b, the public pack, unmodified (a fixed requirement)"],
        ["Quantisation / preset", "int8 weights; lookahead 3, one 320 ms chunk per step"],
        ["Host", "GCP Axion c4a-highcpu-32, 32 x Arm Neoverse-V2, 62 GiB; CPU only"],
        ["Server", "mynah-asr, C11; 6 workers x 5 threads on cpus 0-29; load generator on 30-31"],
        ["Settings", "MYNAH_ASR_STREAM_PAR=4, MYNAH_ASR_FIN_STACK=1 (the 24 Sep qualified settings)"],
        ["Corpora", "FLEURS (CC-BY 4.0): English stress bank (498 clips), French stress bank "
         "(1,143 validated clips), and the frozen EN/FR test bank (200 + 200)"],
        ["Bounds", "the fourteen registered on 22 September, plus two accounting rows added today "
         "(section 3); none relaxed"],
    ], [fw * 0.24, fw * 0.76], header=False))

    # ------------------------------------------------------------------ 2
    A(Paragraph("2. A real waste found: work for clients that had left", H2))
    A(Paragraph(
        "A client that disconnected without a WebSocket close frame was <i>finalized</i> "
        "instead of cancelled: the audio still queued for it &mdash; up to 30 seconds &mdash; "
        "ran through the model for nobody, and the session was counted nowhere. With speech "
        "it was hidden, because the first reply written to a dead socket fails within a write "
        "or two and stops the stream. With silence nothing is written, so nothing failed. The "
        "same waste occurred when a client asked for its final result and then reset the "
        "connection.", BODY))
    A(table([
        ["Case (25 Sep, provoked on purpose)", "Before the fix", "After the fix"],
        ["Client resets mid-utterance, silent audio", "27.4 s computed after it left", "0.00 s"],
        ["Client closes mid-utterance, silent audio", "27.1 s computed after it left", "0.00 s"],
        ["Client asks for the tail, then resets", "27.4 s computed, counted as completed", "0.00 s, counted as a disconnect"],
        ["Frame stalled half-way", "finalized, no error sent, counted nowhere", "idle_timeout"],
        ["Reserved bits / bad control frame", "idle timeout 2 s later", "protocol_error at once"],
    ], [fw * 0.38, fw * 0.34, fw * 0.28]))
    A(Paragraph(
        "Every one of these cases first proves it can see the problem: 27 s of audio were "
        "still queued at the moment of the disconnect, so a zero afterwards means the work "
        "stopped, not that there was none. The legal case &mdash; a client that asks for its "
        "result and then closes its sending side &mdash; still gets its full transcript.", SMALL))

    # ------------------------------------------------------------------ 3
    A(Paragraph("3. Every session accounted for", H2))
    A(Paragraph(
        "The server now keeps exact books: every session it accepts ends exactly once, as "
        "completed, cancelled (with the reason the client was told) or aborted before it "
        "started, and the equation <i>sessions = completed + cancelled + aborted + active</i> "
        "is checked in every snapshot, under load too. The load generator keeps the same books "
        "from its side, and two new verdict rows fail a run if either side loses a single "
        "session. A worker that dies has its sessions counted as lost, never dropped.", BODY))
    A(table([
        ["Fault-injection soaks (25 Sep, 15 min each)", "C64", "C80"],
        ["Clients aborted on purpose, at 7 lifecycle points", "834 / 834 executed", "1,044 / 1,044 executed"],
        ["Sessions: load generator = server", "4,327 = 4,327", "5,390 = 5,390"],
        ["Completed = healthy + tails that finished", "3,551 = 3,493 + 58", "4,420 = 4,346 + 74"],
        ["Idle clients timed out = idle aborts", "123 = 123", "157 = 157"],
        ["Disconnects = hang-ups + reset tails", "653 = 589 + 64", "813 = 736 + 77"],
        ["Errors on the healthy streams", "0", "0"],
        ["Healthy streams: lag p95 / finalization p95", "64 / 99 ms", "77 / 121 ms"],
        ["Transcripts identical to unloaded", "498 of 498", "498 of 498"],
        ["Verdict", "QUALIFIED", "QUALIFIED"],
    ], [fw * 0.46, fw * 0.27, fw * 0.27]))
    A(Paragraph(
        "No evidence of post-disconnect inference: the audio processed was 141 s (C64) and "
        "189 s (C80) lower than the audio the clients sent, because the audio still queued for "
        "an aborted session is discarded rather than computed. The fault suite itself (17 cases "
        "plus a worker killed under a live stream) passes on this machine with the production "
        "model, on a sanitizer build, and with no memory leak after ~70 aborted sessions.", SMALL))

    # ------------------------------------------------------------------ 4
    A(Paragraph("4. What an operator now sees", H2))
    A(Paragraph(
        "In the production layout the metrics endpoint used to show only the router's view "
        "&mdash; connections, admission &mdash; and nothing of what the workers did. Each worker "
        "now publishes its counters to the router, which sums them. The totals never go "
        "backwards when a worker dies, and a test checks that they equal a known mixed "
        "workload exactly. One dashboard panel per question:", BODY))
    A(table([
        ["Question", "Answered by"],
        ["Are requests being served?", "completed sessions and audio seconds per second"],
        ["Are clients disconnecting or failing?", "cancellations by reason; refusals by code"],
        ["Are sessions disappearing or becoming zombies?", "books balanced (alert on 0)"],
        ["Are slots being recovered?", "abandoned minus recovered slots"],
        ["Is latency or backlog deteriorating?", "emission lag, finalization, first-text histograms; worst backlog"],
        ["Did a worker die?", "workers up vs configured; deaths; sessions lost"],
        ["Is the fleet approaching saturation?", "active / slots; model busy time per worker"],
    ], [fw * 0.44, fw * 0.56]))

    # ------------------------------------------------------------------ 5
    A(KeepTogether([Paragraph("5. Capacity: the curve closed", H2), LevelChart(fw, [
        ("EN C128", 30, "GOOD", "24 Sep, x2"),
        ("EN C144", 30, "GOOD", "24 Sep, x2"),
        ("EN C152", 3, "FAILED", "25 Sep, screen"),
        ("EN C160", 2, "FAILED", "24 Sep, screen"),
        ("FR C128", 15, "GOOD", "25 Sep, consistency"),
        ("FR C144", 15, "FAILED", "25 Sep, consistency"),
    ])]))
    A(Paragraph(
        "Bar height is run length, not performance. English 128 and 144 are the 24 September "
        "qualifications (two 30-minute runs each); 152 and 160 are short screens. The French "
        "bars are one 15-minute consistency run each (section 6).", SMALL))
    A(Spacer(1, 6))
    A(table([
        ["Date", "Lang", "Level", "Run", "Utts", "Lag p95", "Final p95", "Backlog", "Lost", "Verdict"],
        ["24 Sep", "EN", "C128", "30 min x2", "34,012", "129 ms", "221 ms", "0.284 s", "0", "QUALIFIED"],
        ["24 Sep", "EN", "C144", "30 min x2", "46,738", "243 ms", "428 ms", "0.464 s", "0", "QUALIFIED"],
        ["25 Sep", "EN", "C152", "3 min", "1,851", "509 ms", "970 ms", "0.704 s", "0", "FAILED"],
        ["24 Sep", "EN", "C160", "2 min", "1,196", "761 ms", "1,338 ms", "1.084 s", "0", "FAILED"],
        ["25 Sep", "FR", "C128", "15 min", "8,133", "153 ms", "261 ms", "0.284 s", "0", "PASS (consistency)"],
        ["25 Sep", "FR", "C144", "15 min", "9,020", "278 ms", "504 ms", "0.384 s", "0", "FAIL (fin 504 > 500)"],
    ], [fw * 0.08, fw * 0.07, fw * 0.07, fw * 0.10, fw * 0.08, fw * 0.09, fw * 0.10, fw * 0.10, fw * 0.06, fw * 0.25]))
    A(Paragraph(
        "Throughput is saturated at about 110 seconds of audio per second at 144, 152 and 160: "
        "past 144, extra concurrency only lengthens the queue. The Lost column is zero on every "
        "row, including the failed ones: the server is late, not lossy.", SMALL))
    A(Spacer(1, 8))
    A(LagChart(fw, {
        "English": ([(128, 129), (144, 243), (152, 509), (160, 761)], ACCENT),
        "French": ([(128, 153), (144, 278)], MARG),
    }))
    A(Paragraph(
        "Emission lag p95 against concurrency, same machine and settings. French sits about "
        "35 ms above English at the same level. Points above 620 ms are drawn at the top of "
        "the axis.", SMALL))

    # ------------------------------------------------------------------ 6
    A(Paragraph("6. French", H2))
    A(Paragraph(
        "<b>The evaluation set was validated before it was used.</b> Of 1,587 French clips "
        "built from FLEURS, 1,143 passed every check and 444 were set aside, each with its "
        "reason: recorded too quietly (306), no language tag emitted by the model (129), "
        "reference without French spelling (35), reference that does not match the audio "
        "(19), English audio (1). Quality was measured on the frozen French test set (200 "
        "clips, never used for tuning).", BODY))
    A(table([
        ["French quality (25 Sep, test set)", "language sent: fr", "language: auto"],
        ["Word error rate, mean [95% interval]", "13.2% [11.5, 15.0]", "13.5% [11.7, 15.4]"],
        ["Character error rate, mean", "6.6%", "6.7%"],
        ["First published word correct", "86.0%", "85.0%"],
        ["Clips better / worse, paired", "14 better, 3 worse, 183 equal", "&mdash;"],
    ], [fw * 0.40, fw * 0.30, fw * 0.30]))
    A(Paragraph(
        "<b>Under load</b> the engine is correct in French at both levels tested &mdash; zero "
        "sessions lost, books balanced, every transcript identical to the idle one. At 128 "
        "streams it passes every bound with wide margins (lag 153 ms, finalization 261 ms). At "
        "144 it missed one bound by 4 ms (finalization 504 against 500), in a single 15-minute "
        "run; the bound was not moved. These are consistency runs, not a new qualification.", BODY))

    # ------------------------------------------------------------------ 7
    A(Paragraph("7. Send the language when you know it", H2))
    A(Paragraph(
        "The model can detect the language itself (<i>auto</i>) or be told it. On the same "
        "model and the same clips, telling it is better in both languages and costs nothing "
        "in latency:", BODY))
    A(table([
        ["Same model, same clips (25 Sep)", "English, 349 clips", "French, 200 clips"],
        ["Word error rate, auto &rarr; explicit", "10.40% &rarr; 10.26%", "13.47% &rarr; 13.20%"],
        ["First words wrong under auto, fixed", "9 of 36, none broken", "2, none broken"],
        ["Latency", "unchanged", "unchanged"],
    ], [fw * 0.40, fw * 0.30, fw * 0.30]))
    A(Paragraph(
        "Several English errors under <i>auto</i> were first words transcribed as another "
        "language. The server already accepts the language on the connection; whether clients "
        "should always send it is a product decision.", SMALL))

    # ------------------------------------------------------------------ 8
    A(Paragraph("8. Where the first word's 2.3 seconds come from", H2))
    A(Paragraph(
        "The first complete word appears 2.3 s after the start of speech at the 95th "
        "percentile. Measured word by word against an independent alignment of the audio, "
        "for the slowest 5% of clips:", BODY))
    A(table([
        ["Part of the 2.33 s p95", "Share", "Whose"],
        ["Before the first word: breath, noise, hesitation", "~1.1 s (41%)", "the recording; the energy detector counts it"],
        ["Saying the first word", "~0.4 s (15%)", "the speaker"],
        ["Model placing the finished word", "~0.65 s", "the model (fixed)"],
        ["Waiting for the next word to close it", "~0.5 s", "the next word's audio"],
        ["The serving runtime", "0 ms", "measured: nothing is held back"],
    ], [fw * 0.46, fw * 0.18, fw * 0.36]))
    A(Paragraph(
        "Measured from the first real word instead of the first sound, the 95th percentile is "
        "1.68 s. Inside the model, the ~0.6 s between the end of a word and its publication "
        "is ~0.4 s where the model itself places the word and ~0.18 s of chunk geometry; the "
        "runtime adds nothing. Serving settings, the lookahead presets and an earlier-commit "
        "rule were all measured and none moves it, so it is recorded as a property of the "
        "model. 37% of the slowest clips also start with a misrecognised first word, most "
        "often a rare name; sending the language fixes some of them (section 7).", SMALL))

    # ------------------------------------------------------------------ 9
    A(Paragraph("9. Scope and names, kept exact", H2))
    A(table([
        ["English qualification", "144 streams, two 30-minute runs (24 Sep). Unchanged by today."],
        ["Fault qualification", "64 and 80 streams, 20% of clients aborted on purpose (25 Sep)."],
        ["French consistency", "one 15-minute run per level: 128 PASS, 144 FAIL by 4 ms. Not a qualification."],
        ["Recommended EN+FR point", "128 streams: a prudent choice, not a new qualified level."],
        ["One host class", "32-core Arm Neoverse-V2. x86 is next."],
        ["Not yet the default", "The 24 Sep settings are qualified but not switched on in the shipped build."],
    ], [fw * 0.26, fw * 0.74], header=False))

    # ------------------------------------------------------------------ 10
    A(Paragraph("10. What comes next", H2))
    A(table([
        ["x86", "The same protocol and settings on an Intel/AMD host, with the scheduler evidence to see "
         "whether the same plateau appears there."],
        ["Fewer pool synchronisations", "The one lever left that could move the ~110x throughput ceiling "
         "rather than the latency under it; after x86."],
        ["Make it the default", "Switch the settings on, choosing 144 (English) or 128 (EN+FR)."],
        ["Worker restart", "Designed, not enabled: a dead worker is counted and routed around today."],
    ], [fw * 0.28, fw * 0.72], header=False))

    # ------------------------------------------------------------------ appendix
    A(Paragraph("Appendix A. Everything done on 25 September &mdash; wins, fails and what is left", H2))
    A(Paragraph(
        "Each item was registered with its rule before it was measured; the failures are "
        "listed with the same weight as the wins.", BODY))
    A(Paragraph("A.1 Work and results", H3))
    A(table([
        ["Item", "What it does", "Measured", "Verdict"],
        ["Harness failure accounting", "No-done replies, warm-up errors, dead client processes and a "
         "schedule artefact become visible", "each gap proved by a planted fault the old code passed", "WIN"],
        ["Cancel on disconnect", "A client gone without a close frame is cancelled, not finalized",
         "27.4 &rarr; 0.00 s computed after it left", "WIN"],
        ["Exact session books", "Every session ends once; client and server reconcile",
         "4,327 = 4,327 and 5,390 = 5,390 under faults", "WIN"],
        ["Fault suite + fault soaks", "17 provoked failures; 20% aborts at 7 points",
         "all green on Arm with the production model; C64 and C80 QUALIFIED", "WIN"],
        ["Service-wide metrics", "Workers publish, the router sums; monotonic across deaths",
         "equal to a known workload exactly", "WIN"],
        ["Stall detector (harness)", "Counted an idle client as a stalled server",
         "found by the fault soaks; now needs queued work", "FIXED"],
        ["French validation", "Validated set, quality, load", "C128 PASS, C144 FAIL by 4 ms", "PASS at 128"],
        ["Language sent explicitly", "fr / en instead of auto, same model",
         "better WER and first word in both, no regressions", "WIN"],
        ["C152 screen", "The level between 144 and 160", "lag 509, fin 970 ms; throughput flat", "FAIL"],
        ["Lookahead presets for first word", "80 ms to 1,120 ms chunks",
         "first-word p95 unchanged; the smallest costs +1 WER point", "FAIL"],
        ["First-word attribution", "Where the 2.3 s go", "runtime 0 ms; model ~0.6 s; lead-in ~1.1 s", "ANSWERED"],
    ], [fw * 0.20, fw * 0.32, fw * 0.32, fw * 0.16], size=7.8))
    A(Paragraph("A.2 What the failures taught", H3))
    A(table([
        ["Speech hides a zombie", "A reply to a dead client fails and stops the work; silence writes "
         "nothing. The test that found it sends silence on purpose."],
        ["A test must prove it can fail", "Every zombie case checks that work was still queued; on a fast "
         "machine the first version passed without testing anything."],
        ["Platforms differ in how a reset looks", "macOS and Linux report a reset differently; one "
         "operating-system quirk seen on the development machine does not occur on the server."],
        ["An idle client is not a stalled server", "The stall bound failed both fault soaks on one idle "
         "client each; the bound now requires queued work."],
        ["4 ms is a fail", "French at 144 missed finalization by 0.8%; the bound was not moved."],
    ], [fw * 0.26, fw * 0.74], header=False, size=8.2))
    A(KeepTogether([Paragraph("A.3 Not done yet", H3), table([
        ["x86 campaign", "The next item: same protocol, same settings, a second architecture."],
        ["Fewer pool synchronisations", "~9,700 pool dispatches per second per worker at the new settings."],
        ["Make the settings the default", "A product decision: 144 (English) or 128 (EN+FR)."],
        ["French at 144, two runs", "Only if 144 is ever needed for French."],
        ["Worker restart", "Designed with crash-loop protection; off until it has its own fault test."],
        ["Other languages", "The pack serves 40; English and French are the ones validated."],
    ], [fw * 0.30, fw * 0.70], header=False, size=8.2)]))

    # ------------------------------------------------------------------ 11
    A(Paragraph("11. Reproducing this", H2))
    A(Paragraph(
        "make test-server-faults FAULT_MODEL_DIR=models/nemotron-3.5-asr-streaming-0.6b<br/>"
        "MYNAH_ASR_STREAM_PAR=4 MYNAH_ASR_FIN_STACK=1 tools/bench/v2_qualify.sh \\<br/>"
        "&nbsp;&nbsp;-m models/nemotron-3.5-asr-streaming-0.6b --phase soak --soak-c 64 \\<br/>"
        "&nbsp;&nbsp;--soak-seconds 900 --soaks 1 --abort-pct 20 --abort-seed 7 [--lang fr] \\<br/>"
        "&nbsp;&nbsp;--corpus samples/stress-fr/manifest-validated.json --corpus-sample 500 \\<br/>"
        "&nbsp;&nbsp;-W 6 -T 5 -C 96 --http-threads 32 --server-cpus 0-29 --gen-cpus 30-31<br/>"
        "python3 tools/bench/v2_verdict.py &lt;run&gt;", MONO))
    A(Spacer(1, 6))
    A(Paragraph(
        "Measurements taken 25 September 2026 on GCP Axion c4a-highcpu-32. Model "
        "nvidia/nemotron-3.5-asr-streaming-0.6b, int8, preset [56, 3]. Corpora FLEURS "
        "English and French (CC-BY 4.0). Runtime mynah-asr, C11, research branch at commit "
        "ac21e26 for the French runs and 2aee1be for the fault runs. Prepared by Gabriele "
        "Mastrapasqua.", SMALL))
    doc.build(F)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "asr-report-20260925.pdf")
