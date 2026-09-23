#!/usr/bin/env python3
"""Nemotron streaming ASR on Arm - the client-facing PDF. Generated with reportlab.

Lives in tools/ and not in .work/ because .work/evidence is untracked by design:
the artefacts it reads are benchmark dumps that must never be committed, but this
generator is source for a deliverable. Data in, code tracked.

The visual language is deliberately the same as the PocketTTS report of
2026-09-21 (mynah-tts tools/client_report_pocket.py): same palette, same band,
same table and chart idiom, so a reader who has seen one can read the other
without relearning it. The content is entirely different -- that document
measures synthesis, this one measures recognition.

    uv run --with reportlab python3 tools/client_report_axion_asr.py OUT.pdf
"""
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, KeepTogether,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

INK    = colors.HexColor("#1a1a1a")
MUTE   = colors.HexColor("#6b6b6b")
RULE   = colors.HexColor("#d8d8d8")
GOOD   = colors.HexColor("#2f7d4f")
MARG   = colors.HexColor("#b07d1c")
BAD    = colors.HexColor("#a33a2e")
BAND   = colors.HexColor("#f4f4f2")
ACCENT = colors.HexColor("#1f4e79")

S = getSampleStyleSheet()
def st(name, **kw):
    base = dict(fontName="Helvetica", fontSize=9.5, leading=14, textColor=INK)
    base.update(kw)
    return ParagraphStyle(name, **base)

BODY  = st("body", spaceAfter=6)
SMALL = st("small", fontSize=8.2, leading=11.5, textColor=MUTE)
H1    = st("h1", fontName="Helvetica-Bold", fontSize=17, leading=21, spaceAfter=2)
H2    = st("h2", fontName="Helvetica-Bold", fontSize=11.5, leading=15,
           spaceBefore=13, spaceAfter=5, textColor=ACCENT)
H3    = st("h3", fontName="Helvetica-Bold", fontSize=9.5, leading=13,
           spaceBefore=8, spaceAfter=3)
SUB   = st("sub", fontSize=10, leading=14, textColor=MUTE)
MONO  = st("mono", fontName="Courier", fontSize=8, leading=11)
CELL  = st("cell", fontSize=8.6, leading=11.4)
CELLB = st("cellb", fontName="Helvetica-Bold", fontSize=8.6, leading=11.4)


def _wrap(rows, header):
    out = []
    for r, row in enumerate(rows):
        style = CELLB if (header and r == 0) else CELL
        out.append([c if not isinstance(c, str) else Paragraph(c, style) for c in row])
    return out


def table(rows, widths, header=True, align=None, size=8.6):
    t = Table(_wrap(rows, header), colWidths=widths, hAlign="LEFT")
    cmds = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", size),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, RULE),
    ]
    if header:
        cmds += [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", size),
                 ("BACKGROUND", (0, 0), (-1, 0), BAND),
                 ("LINEBELOW", (0, 0), (-1, 0), 0.6, MUTE)]
    for col, a in (align or {}).items():
        cmds.append(("ALIGN", (col, 0), (col, -1), a))
        cmds.append(("RIGHTPADDING", (col, 0), (col, -1), 6))
    t.setStyle(TableStyle(cmds))
    return t


class Rule(Flowable):
    def __init__(self, w, thick=0.6, color=RULE, pad=3):
        self.width, self.thick, self.color, self.pad = w, thick, color, pad
        self.height = thick + pad * 2
    def draw(self):
        self.canv.setStrokeColor(self.color); self.canv.setLineWidth(self.thick)
        self.canv.line(0, self.pad, self.width, self.pad)


class Callout(Flowable):
    def __init__(self, width, items):
        self.width, self.items = width, items
        self.height = 30 * mm
    def draw(self):
        c = self.canv
        c.setFillColor(BAND); c.rect(0, 0, self.width, self.height, stroke=0, fill=1)
        c.setStrokeColor(ACCENT); c.setLineWidth(2); c.line(0, 0, 0, self.height)
        col = self.width / len(self.items)
        for i, (big, small) in enumerate(self.items):
            x = 8 + i * col
            c.setFillColor(ACCENT); c.setFont("Helvetica-Bold", 19)
            c.drawString(x, self.height - 15 * mm, big)
            c.setFillColor(MUTE); c.setFont("Helvetica", 7.6)
            for j, line in enumerate(small):
                c.drawString(x, self.height - 20 * mm - j * 9, line)


class LevelChart(Flowable):
    """Every concurrency level, how long it was held, and the verdict.

    The bar height is run LENGTH, not performance: it is there to make the
    difference between a three-minute screen and a thirty-minute qualification
    impossible to miss, because only the second kind promotes anything."""
    def __init__(self, width, data):
        self.width, self.data = width, data
        self.height = 66 * mm
    def draw(self):
        c = self.canv
        L, R, B = 26, 12, 30
        plot_w = self.width - L - R
        plot_h = self.height - B - 20
        c.setFont("Helvetica-Bold", 8.5); c.setFillColor(INK)
        c.drawString(0, self.height - 8,
                     "Concurrency levels measured, run length and verdict")
        c.setFont("Helvetica", 7); c.setFillColor(MUTE)
        for m in (0, 10, 20, 30):
            y = B + plot_h * m / 32.0
            c.setStrokeColor(RULE); c.setLineWidth(0.3); c.line(L, y, L + plot_w, y)
            c.drawRightString(L - 4, y - 2.5, "%d" % m)
        c.saveState(); c.translate(7, B + plot_h / 2); c.rotate(90)
        c.setFont("Helvetica", 7.2); c.drawCentredString(0, 0, "run minutes")
        c.restoreState()
        slot = plot_w / len(self.data)
        for i, (label, mins, verdict, note) in enumerate(self.data):
            x = L + i * slot + slot * 0.22
            w = slot * 0.56
            h = plot_h * mins / 32.0
            col = {"GOOD": GOOD, "MARGINAL": MARG, "FAILED": BAD}[verdict]
            c.setFillColor(col); c.rect(x, B, w, h, stroke=0, fill=1)
            c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 7)
            if h > 14:
                c.drawCentredString(x + w / 2, B + h - 9,
                                    "%d min" % mins if mins >= 1 else "")
            c.setFillColor(col); c.setFont("Helvetica-Bold", 7.2)
            c.drawCentredString(x + w / 2, B + h + 4, verdict)
            c.setFillColor(INK); c.setFont("Helvetica-Bold", 8.4)
            c.drawCentredString(x + w / 2, B - 11, label)
            c.setFillColor(MUTE); c.setFont("Helvetica", 6.4)
            c.drawCentredString(x + w / 2, B - 19, note)
        c.setStrokeColor(MUTE); c.setLineWidth(0.6); c.line(L, B, L + plot_w, B)


class LagChart(Flowable):
    """Emission lag against the cadence the model itself sets."""
    def __init__(self, width, series, bound=320.0):
        self.width, self.series, self.bound = width, series, bound
        self.height = 54 * mm
    def draw(self):
        c = self.canv
        L, R, B = 32, 14, 26
        pw = self.width - L - R
        ph = self.height - B - 18
        lo, hi = 0.0, 620.0
        ypix = lambda v: B + ph * (v - lo) / (hi - lo)
        xs = [p[0] for pts, _ in self.series.values() for p in pts]
        xlo, xhi = min(xs) - 4, max(xs) + 4
        xpix = lambda v: L + pw * (v - xlo) / (xhi - xlo)
        c.setFont("Helvetica-Bold", 8.5); c.setFillColor(INK)
        c.drawString(0, self.height - 8,
                     "Emission lag, 95th percentile, against the 320 ms cadence")
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
        c.drawRightString(L + pw, yb + 5, "320 ms bound  (lookahead+1) x 80")
        for name, (pts, col) in self.series.items():
            pts = sorted(pts, key=lambda p: p[0])
            for i in range(1, len(pts)):
                c.setStrokeColor(col); c.setLineWidth(1.0)
                c.line(xpix(pts[i - 1][0]), ypix(min(pts[i - 1][1], 615)),
                       xpix(pts[i][0]), ypix(min(pts[i][1], 615)))
            for cc, val in pts:
                x, y = xpix(cc), ypix(min(val, 615))
                c.setFillColor(colors.white); c.circle(x, y, 3.8, stroke=0, fill=1)
                c.setFillColor(col); c.circle(x, y, 2.7, stroke=0, fill=1)
            lx, ly = xpix(pts[-1][0]), ypix(min(pts[-1][1], 615))
            c.setFillColor(col); c.setFont("Helvetica-Bold", 7)
            c.drawString(lx + 6, ly - 2, name)
        c.setFillColor(INK); c.setFont("Helvetica-Bold", 7.6)
        for cc in sorted({p[0] for pts, _ in self.series.values() for p in pts}):
            c.drawCentredString(xpix(cc), B - 10, "C%d" % cc)
        c.setStrokeColor(MUTE); c.setLineWidth(0.6); c.line(L, B, L + pw, B)


def build(path):
    doc = BaseDocTemplate(path, pagesize=A4,
                          leftMargin=20 * mm, rightMargin=18 * mm,
                          topMargin=17 * mm, bottomMargin=17 * mm,
                          title="Nemotron streaming ASR on Arm - concurrency report",
                          author="Gabriele Mastrapasqua")
    fw = doc.width
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="n")

    def decorate(canv, d):
        canv.saveState()
        canv.setFont("Helvetica", 7.2); canv.setFillColor(MUTE)
        canv.drawString(doc.leftMargin, 11 * mm,
                        "Streaming ASR on Arm - GCP Axion c4a - 23 September 2026")
        canv.drawRightString(doc.leftMargin + doc.width, 11 * mm, "page %d" % d.page)
        canv.setStrokeColor(RULE); canv.setLineWidth(0.4)
        canv.line(doc.leftMargin, 13.5 * mm, doc.leftMargin + doc.width, 13.5 * mm)
        canv.restoreState()

    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=decorate)])
    F = []
    A = F.append

    # ------------------------------------------------------------------ title
    A(Paragraph("Streaming speech recognition on Arm", H1))
    A(Paragraph("Concurrent real-time ASR on a single 32-core Arm server &mdash; "
                "GCP Axion c4a &middot; 22-23 September 2026", SUB))
    A(Spacer(1, 4)); A(Rule(fw, 1.2, ACCENT)); A(Spacer(1, 8))

    A(Callout(fw, [
        ("80", ["concurrent live streams", "qualified over 2 x 30 minutes"]),
        ("0", ["streams lost in 21,287", "utterances, 78 hours of audio"]),
        ("118 ms", ["emission lag, 95th percentile", "against a 320 ms bound"]),
        ("77x", ["realtime aggregate", "audio seconds per second"]),
    ]))
    A(Spacer(1, 10))

    A(Paragraph(
        "One 32-core Arm virtual machine, CPU only, no GPU and no Python in the "
        "serving path, transcribes 80 simultaneous live audio streams for thirty "
        "minutes without losing a stream, stalling, or changing a single word it "
        "would have said with the machine idle. This document states what was "
        "measured, what a level has to survive before it may be called qualified, "
        "and what has not been measured yet.", LEAD := st("lead", fontSize=11, leading=16)))

    A(Spacer(1, 6))
    A(Paragraph("Where it started and where it is now", H3))
    A(table([
        ["", "22 Sep", "23 Sep", "Change"],
        ["Qualified concurrency", "16", "80", "+400%"],
        ["Server layout", "3 workers x 8 threads, 24 cpus", "6 workers x 5 threads, 30 cpus", "&mdash;"],
        ["Aggregate throughput", "15.6x realtime", "77x realtime", "+394%"],
        ["Emission lag, p95", "66 ms", "118 ms", "at 5x the load"],
        ["Streams lost", "0", "0", "&mdash;"],
        ["Corpus", "27 clips", "498 clips, 3.2&ndash;42 s", "&mdash;"],
    ], [fw * 0.26, fw * 0.24, fw * 0.28, fw * 0.22]))
    A(Paragraph(
        "Same virtual machine, same model, same build. Nothing in this table comes "
        "from better hardware, a smaller model or a looser standard: the bounds on "
        "23 September are the ones registered on 22 September plus two more. What "
        "changed is described in section 5.", SMALL))

    # ------------------------------------------------------------------- 1
    A(Paragraph("1. What is being measured", H2))
    A(Paragraph("The model is NVIDIA Nemotron 3.5 streaming ASR, the public 0.6B "
                "cache-aware pack, used unmodified.", BODY))
    A(table([
        ["Model", "nvidia/nemotron-3.5-asr-streaming-0.6b"],
        ["Architecture", "FastConformer encoder with an RNN-T decoder, 0.6B parameters, "
                         "cache-aware streaming"],
        ["Quantisation", "int8 weights, the shipped default, nothing hand-tuned for this test"],
        ["Preset", "lookahead 3 &mdash; the encoder advances one 320 ms chunk per step "
                   "(<font face='Courier'>(lookahead+1) x 80 ms</font>)"],
        ["Audio", "16 kHz mono, fed to the server in 100 ms frames at real time"],
        ["Languages", "English measured here; the pack serves 40"],
    ], [fw * 0.20, fw * 0.80]))
    A(Spacer(1, 4))
    A(Paragraph("The runtime is mynah-asr, a C11 inference engine. Weights are "
                "memory-mapped, the matrix kernels are its own, and this host runs "
                "with <font face='Courier'>BLAS=none</font>, so no external BLAS is "
                "involved and no Python runs while a stream is being served.", BODY))
    A(table([
        ["Host", "GCP Axion c4a-highcpu-32, 32 x Arm Neoverse-V2, 62 GiB, Ubuntu, gcc 15.2"],
        ["Build", "BLAS=none, SIMD neon+dotprod+i8mm, CPU only. int8 matrices resolve to "
                  "<font face='Courier'>neon-smmla</font>, verified from the binary, not assumed"],
        ["Server", "6 worker processes x 5 threads, pinned to cpus 0-29, 96 stream slots each"],
        ["Load generator", "pinned to cpus 30-31, off the server's cpus entirely"],
        ["Commit", "<font face='Courier'>a969877</font>, clean tree; binary sha256 "
                   "<font face='Courier'>551cfd94...</font>"],
    ], [fw * 0.20, fw * 0.80]))

    # ------------------------------------------------------------------- 2
    A(Paragraph("2. What \u201cqualified\u201d means here", H2))
    A(Paragraph(
        "A recognition server can be fast on average and still be unusable, because "
        "a listener does not experience an average &mdash; they experience the one "
        "moment the words stopped arriving, or the one request whose transcript came "
        "back different because the machine was busy. Fourteen bounds were registered "
        "<b>before the first measured run</b>, and a level is judged against all of "
        "them.", BODY))
    A(table([
        ["Bound", "Threshold", "What it protects"],
        ["Established streams lost", "0", "A stream accepted and then dropped is a product "
                                          "failure at any concurrency"],
        ["Emission lag, p95", "&le; 320 ms", "The cadence the model itself sets. Past it a "
                                             "stream is falling behind the audio"],
        ["Finalization, p95", "&le; 500 ms", "The wait after the speaker stops"],
        ["Backlog, max", "&le; 0.64 s", "Audio received and not yet encoded"],
        ["Per-window drift", "every 60 s window inside the bound",
         "A whole-run percentile hides a bad minute. Fail at minute 23 and the run failed"],
        ["Trend", "last third &le; 1.5x first third", "A soak slowly getting worse has not qualified"],
        ["Server-side stall", "0 intervals with work and no progress",
         "Read from the fleet's own dumps, not from an absence of complaints"],
        ["Client-observable stall", "max lag &le; 3 s", "The maximum, not a percentile. "
         "\u201cEventually recovered\u201d is still a stall"],
        ["Transcript parity", "byte-identical to the unloaded transcript",
         "Load may not change what the server says"],
        ["Worker RSS", "&le; 1.15x", "A leak over thirty minutes is a leak"],
        ["Worker deaths", "0", "A fleet that silently replaces a worker has not held the load"],
        ["TTFP load penalty, p95", "&le; 250 ms", "Paired per clip against its own unloaded "
         "time, so the corpus's silence cancels"],
        ["Ready to first partial, p95", "&le; one cadence", "Queueing before the first word"],
    ], [fw * 0.22, fw * 0.22, fw * 0.56]))
    A(Spacer(1, 4))
    A(Paragraph(
        "Two rules of method are worth stating, because they are what make the "
        "headline number trustworthy.", BODY))
    A(Paragraph(
        "<b>Discovery and qualification are different jobs.</b> Short runs find where "
        "the machine breaks; they may disqualify a level and they never promote one. "
        "Only a thirty-minute run on a fresh fleet promotes, and two of them with "
        "different random schedules, so a result cannot be one lucky ordering.", BODY))
    A(Paragraph(
        "<b>The corpus is a population, not a demo.</b> 498 English clips from FLEURS, "
        "3.2 to 42.4 seconds, in three length classes, sampled deterministically from "
        "a bank of 1,587. In the qualifying run each clip was spoken about 21 times. "
        "The earlier 16-stream figure used 27 clips, each played 145 times, which "
        "measures a warm cache as much as a fleet.", BODY))

    # ------------------------------------------------------------------- 3
    # KeepTogether so the heading cannot end a page alone with its chart overleaf
    A(KeepTogether([Paragraph("3. Results", H2), LevelChart(fw, [
        ("C16", 30, "GOOD", "22 Sep, 3x8"),
        ("C80", 30, "GOOD", "QUALIFIED x2"),
        ("C88", 3, "GOOD", "screen"),
        ("C96", 3, "GOOD", "screen, last"),
        ("C104", 3, "FAILED", "3 bounds"),
        ("C120", 3, "FAILED", "overloaded"),
    ])]))
    A(Paragraph(
        "Green cleared every bound. Only the thirty-minute runs are allowed to "
        "promote a level; the three-minute bars are screens and are drawn short on "
        "purpose. C96 cleared every bound on a screen and was <b>not</b> chosen: it "
        "runs the model 97% of the time, which leaves nothing for a bad day. C104 "
        "failed finalization, backlog and per-window drift together &mdash; a capacity "
        "limit, not one threshold moving in noise.", SMALL))
    A(Spacer(1, 6))
    A(table([
        ["Level", "Run", "Utterances", "Lag p95", "Backlog", "Final p95", "Lost", "Verdict"],
        ["C64", "3 min", "806", "80 ms", "0.264 s", "160 ms", "0", "GOOD (screen)"],
        ["C80", "3 min", "1,005", "108 ms", "0.284 s", "221 ms", "0", "GOOD (screen)"],
        ["C80", "30 min", "10,647", "115 ms", "0.384 s", "224 ms", "0", "QUALIFIED"],
        ["C80", "30 min", "10,640", "118 ms", "0.384 s", "231 ms", "0", "QUALIFIED"],
        ["C88", "3 min", "1,109", "133 ms", "0.384 s", "271 ms", "0", "GOOD (screen)"],
        ["C96", "3 min", "1,515", "182 ms", "0.484 s", "379 ms", "0", "GOOD (screen)"],
        ["C104", "3 min", "1,285", "305 ms", "0.744 s", "619 ms", "0", "FAILED"],
        ["C112", "3 min", "1,348", "556 ms", "0.944 s", "1081 ms", "0", "FAILED"],
        ["C120", "3 min", "1,669", "1328 ms", "2.224 s", "2058 ms", "0", "FAILED"],
    ], [fw * 0.09, fw * 0.10, fw * 0.14, fw * 0.12, fw * 0.12, fw * 0.13, fw * 0.08, fw * 0.22],
        align={2: "RIGHT", 3: "RIGHT", 4: "RIGHT", 5: "RIGHT", 6: "RIGHT"}))
    A(Paragraph(
        "The <b>Lost</b> column is the one to read twice. It is zero on every row, "
        "including the rows that failed: at 120 streams, with text arriving 1.3 "
        "seconds behind the speech, the server had still not dropped a stream it had "
        "accepted. Under overload this fleet gets slow. It does not lose work and it "
        "does not corrupt it.", SMALL))
    A(Spacer(1, 8))
    A(LagChart(fw, {
        "6x5, 30 cpus": ([(64, 80), (80, 108), (88, 133), (96, 182), (104, 305),
                          (112, 556), (120, 1328)], ACCENT),
        "3x8, 24 cpus": ([(32, 59), (40, 78), (48, 106), (56, 131), (64, 207),
                          (72, 546), (80, 2064)], MARG),
    }))
    A(Paragraph(
        "Both layouts run the same binary, model and corpus on the same machine. The "
        "24-cpu layout leaves the bound between 64 and 72 streams; the 30-cpu layout "
        "is still inside it at 96. Points above 620 ms are drawn at the top of the "
        "axis.", SMALL))

    # ------------------------------------------------------------------- 4
    A(Paragraph("4. The operating point in detail", H2))
    A(Paragraph("C80, two independent thirty-minute runs on a fresh fleet each, "
                "different random schedules, shipped configuration:", BODY))
    A(table([
        ["Utterances completed", "21,287 across the two runs (none dropped, none timed out)"],
        ["Audio transcribed", "281,821 seconds &mdash; 78.3 hours"],
        ["Established streams lost", "0"],
        ["Admission refusals (503)", "0"],
        ["Emission lag", "p50 47 ms, p95 118 ms, p99 173 ms, max 560 ms"],
        ["Worst 60-second window", "124 ms p95, against a 320 ms bound"],
        ["Finalization", "p95 231 ms, p99 301 ms"],
        ["Backlog, max", "0.384 s (bound 0.64 s)"],
        ["Stalls", "0 server-side over 726 sampled intervals; of 585,693 published "
                   "partials, 95 crossed one cadence and <b>none</b> crossed two"],
        ["Drift", "+6.1% and +0.1% across the run; no window outside the bound"],
        ["Memory", "1.037x peak growth over thirty minutes, no worker replaced"],
        ["Fairness", "worst stream's p95 is 1.18x the median stream's, over 80 streams"],
        ["Transcript parity", "498 of 498 clips byte-identical to the unloaded transcript"],
        ["Aggregate throughput", "77 seconds of speech transcribed per second of wall clock"],
        ["CPU actually used", "21.6 and 22.0 cores of the 30 allocated, measured from "
                              "cumulative cpu-seconds"],
    ], [fw * 0.27, fw * 0.73]))
    A(Paragraph(
        "In plainer terms: the machine transcribes about an hour and twenty minutes "
        "of speech per minute of its own time, eighty people can be talking to it at "
        "once, and each of them sees their words appear about a tenth of a second "
        "behind the audio.", BODY))

    # ------------------------------------------------------------------- 5
    A(Paragraph("5. Why the number moved from 16 to 80", H2))
    A(Paragraph(
        "A five-fold capacity increase on unchanged hardware and an unchanged model "
        "deserves an explanation rather than a celebration. None of it came from "
        "making recognition worse &mdash; the transcripts are byte-identical to the "
        "unloaded ones at every level measured. Two of the three changes cost nothing "
        "at all, and the third was a measurement that contradicted what we had "
        "written down.", BODY))
    A(Paragraph("5.1 The first 16 was never a limit", H3))
    A(Paragraph(
        "C16 was a floor we had been asked to guarantee, not a ceiling the machine "
        "had reached. At 16 streams the fleet ran the model 30% of the time. The "
        "honest reading of an idle-looking server was that nobody had yet asked it "
        "for much.", BODY))
    A(Paragraph("5.2 A benchmark that measured its own harness", H3))
    A(Paragraph(
        "The server's HTTP thread pool is also its connection ceiling &mdash; a "
        "WebSocket stream holds one thread for its whole life. The pool was set to 8 "
        "per worker, so a three-worker fleet could hold 24 streams while advertising "
        "96 slots. A run asked for 32 and served 24, with no error and no refusal, "
        "because a stream that never connects is neither. It showed up only as "
        "<i>fewer</i> utterances at C32 than at C24, which reads like noise. The tool "
        "now reads the pool back from the worker's own start-up banner and refuses a "
        "level above the fleet's ceiling.", BODY))
    A(Paragraph("5.3 Eight cores were reserved for a client that needed 0.09", H3))
    A(Paragraph(
        "Eight of the 32 cores were withheld from the server so the load generator "
        "could not compete with it. Measured, the generator used a tenth of one core "
        "to drive 32 streams, and gave identical results on 8, 4 and 2 cores &mdash; "
        "with the 8-core arm showing the <i>worst</i> send lateness of the three, "
        "which is noise. Six cores were recovered and deliberately left empty until a "
        "separate experiment established what a topology could do with them.", BODY))
    A(Paragraph("5.4 The layout, not the core count", H3))
    A(table([
        ["Layout", "cpus", "Throughput at C64", "Lag p95 at C64", "Highest level cleared"],
        ["3 x 8", "24", "53.7x", "190 ms", "C56"],
        ["3 x 10", "30", "53.7x", "183 ms", "C56"],
        ["2 x 15", "30", "46.3x", "2007 ms", "none"],
        ["5 x 6", "30", "54.4x", "87 ms", "C88"],
        ["6 x 5", "30", "54.4x", "80 ms", "<b>C96</b>"],
    ], [fw * 0.14, fw * 0.10, fw * 0.22, fw * 0.20, fw * 0.34],
        align={2: "RIGHT", 3: "RIGHT"}))
    A(Paragraph(
        "Giving the same three workers two more threads each bought nothing: 53.7x "
        "against 53.7x. Giving two workers fifteen threads each made it four times "
        "worse. Splitting the same 30 cpus into six narrow workers moved the last "
        "good level from 56 to 96. The gain is more independent domains, not more "
        "threads inside one.", SMALL))
    A(Paragraph(
        "<b>What this does not establish.</b> Why narrow-and-many wins is not proved "
        "here, and these measurements do not locate a general limit on threads per "
        "worker: they show that 15 is bad and that 10 buys nothing over 8 for this "
        "model on this machine. At the qualified level the fleet uses 22 of its 30 "
        "cores, and why the other 8 core-equivalents go unused &mdash; and why that "
        "number falls further under overload &mdash; is an open question, not a "
        "finding.", BODY))

    # ------------------------------------------------------------------- 6
    A(Paragraph("6. What to run on this machine", H2))
    A(table([
        ["Recommended concurrency", "80 simultaneous streams",
         "Qualified: 2 x 30 minutes, 21,287 utterances, zero losses, zero stalls"],
        ["Server layout", "6 workers x 5 threads, cpus 0-29",
         "96 stream slots each; measured better than 3x8, 3x10, 2x15 and 5x6"],
        ["Expected emission lag", "118 ms at the 95th percentile",
         "Bound is 320 ms, the model's own chunk cadence"],
        ["Expected throughput", "77x realtime", "77 seconds of speech per second of wall clock"],
        ["Highest level screened clean", "96 streams", "Three minutes, not thirty. Not promised"],
        ["First level that failed", "104 streams", "Finalization, backlog and drift together"],
        ["Headroom above the recommendation", "30%",
         "104 / 80. Chosen over C88, which is 18% below failure at 92% model duty"],
    ], [fw * 0.24, fw * 0.24, fw * 0.52]))
    A(Paragraph(
        "<b>Sizing by demand.</b> At 80 streams per machine, a service expecting a "
        "peak of 1,000 concurrent speakers needs 13 machines of this class; 2,500 "
        "needs 32. Two of the 32 cores were held back for the benchmark client, which "
        "in a real deployment does not live on the inference host &mdash; so the "
        "qualified figure is conservative by whatever those two cores are worth, and "
        "that has not been measured.", BODY))

    # ------------------------------------------------------------------- 7
    A(Paragraph("7. How long before the first word appears", H2))
    A(Paragraph(
        "This is the number a user actually feels, and it is the one place where this "
        "report has to be careful, because the obvious measurement is misleading. Time "
        "from the stream opening to the first published word mixes three unrelated "
        "things: the silence the recording begins with, the speech this model wants "
        "before it will commit to a word, and the delay the server adds. On this "
        "corpus the first two dominate.", BODY))
    A(table([
        ["What", "Value", "Who owns it"],
        ["Stream open to first word, p95", "2,845 ms", "mostly the recording"],
        ["<b>Speech onset</b> to first word, p50", "867 ms", "the checkpoint"],
        ["<b>Speech onset</b> to first word, p95", "1,969 ms", "the checkpoint and this corpus"],
        ["<b>Load penalty</b>, p95, paired per clip", "<b>+86 ms</b>", "<b>the server</b>"],
        ["Queueing before the first partial, p95", "+86 ms", "the server"],
    ], [fw * 0.40, fw * 0.24, fw * 0.36], align={1: "RIGHT"}))
    A(Paragraph(
        "The fourth row is the only one a serving change can move, and it is the one "
        "that is gated. It is measured per clip against that same clip's transcript "
        "with the machine idle, so whatever silence a recording begins with appears "
        "on both sides and cancels.", SMALL))
    A(Paragraph(
        "<b>Stated plainly, because it is a real product figure:</b> at 80 streams a "
        "speaker waits about two seconds at the 95th percentile between starting to "
        "talk and seeing their first word. Of those two seconds the server "
        "contributes 86 milliseconds. The rest belongs to the model and to this "
        "corpus, whose long class runs to 42 seconds; the same build on shorter, "
        "cleaner clips reads 1,251 ms. Against our own responsiveness guideline of "
        "1,500 ms this is outside target, it is recorded as such, and it will not be "
        "improved by serving work.", BODY))

    # ------------------------------------------------------------------- 8
    A(Paragraph("8. Recognition quality, and what load did to it", H2))
    A(Paragraph(
        "Two different questions, kept apart on purpose.", BODY))
    A(table([
        ["Serving correctness", "498 of 498 clips byte-identical to the transcript the "
         "same build produced with the machine idle, in both thirty-minute runs. Zero "
         "regressions across 476 sampled utterances in the archive.",
         "<b>This is what the server is judged on.</b>"],
        ["Model quality", "WER 7.1% median, CER 2.7% median against the human reference, "
         "over 21,287 utterances of FLEURS English.",
         "Reported, not gated. A word this checkpoint gets wrong on an idle machine is "
         "not a defect of the fleet, and no serving work changes it."],
    ], [fw * 0.18, fw * 0.50, fw * 0.32], header=False))
    A(Paragraph(
        "The distinction matters for reading the archive: a clip where the transcript "
        "differs from the human reference is the model's error and is visible in both "
        "the loaded and the unloaded column. A clip where the two columns differ from "
        "<i>each other</i> would be a serving defect. There are none.", SMALL))

    # ------------------------------------------------------------------- 9
    A(Paragraph("9. The audio in the accompanying archive", H2))
    A(Paragraph(
        "Every clip in the archive was transcribed by the runtime described here, "
        "captured from the live server while it was carrying all 80 concurrent "
        "streams &mdash; not produced quietly on an idle machine.", BODY))
    A(table([
        ["Folder", "What it contains"],
        ["under-load/", "Clips drawn from the qualifying run, each with the audio "
                        "exactly as it was sent, the human reference, the transcript "
                        "the server produced with the machine idle, the transcript it "
                        "produced under full load, and the sequence of partial results "
                        "with the time each one arrived"],
        ["index.jsonl", "One row per utterance: clip, reference, unloaded transcript, "
                        "loaded transcript, partials with timestamps, onset, TTFP and "
                        "its paired load penalty, first-delta lag, finalization, WER "
                        "and CER against the human reference, and whether it was a "
                        "serving regression"],
        ["TRANSCRIPTS.txt", "The same thing as plain text, for reading without tools"],
    ], [fw * 0.22, fw * 0.78]))
    A(Paragraph(
        "The archive holds every utterance that went wrong &mdash; errors, refusals, "
        "empty output, any transcript that diverged from its unloaded reference &mdash; "
        "plus a deterministic, length-balanced sample of ordinary ones. It is not a "
        "selection of the good cases: the failure cases are all of them, and there "
        "were no serving regressions to include.", SMALL))

    # ------------------------------------------------------------------ 10
    A(Paragraph("10. Scope of these numbers", H2))
    A(table([
        ["One language, one model", "English Nemotron 0.6B streaming only. The pack "
         "serves 40 languages and none of the others has been through this campaign."],
        ["One host class", "A 32-core Arm machine. Nothing here predicts a 16-core or "
         "64-core host: the right worker layout follows the model's own cost structure "
         "and has to be re-screened per host. On this machine the layout mattered more "
         "than the core count."],
        ["Arm measured; x86 next", "Every number is from Arm Neoverse-V2. The x86 build "
         "exists and passes the same correctness suites in CI on every change, and "
         "carries AVX2 and VNNI code paths; it has not been through this concurrency "
         "campaign."],
        ["The load generator shares the box", "The client that measured these numbers "
         "ran on two cores of the same machine. That makes the figure conservative "
         "rather than optimistic."],
        ["The exact figure between 96 and 104", "96 cleared a three-minute screen and "
         "104 failed one. 80 is qualified over thirty minutes twice. The levels "
         "between have not been held for thirty minutes and are not claimed."],
        ["First-word latency is not certified", "Section 7. The serving contribution is "
         "gated and passes; the total wait is a property of the checkpoint and the "
         "corpus and is reported, not promised."],
    ], [fw * 0.26, fw * 0.74], header=False))

    # ------------------------------------------------------------------ 11
    A(Paragraph("11. What comes next", H2))
    A(table([
        ["Qualify a level between 80 and 96", "96 screens clean. One thirty-minute pair "
         "at 88 would say whether the recommendation can move up, and would cost an hour."],
        ["Explain the unused cores", "The fleet uses 22 of 30 cores at the qualified "
         "level and fewer under overload. Whatever that is, it is the next capacity "
         "increase and it is not new hardware."],
        ["Batching at saturation", "Almost every model step still serves a single "
         "stream. Whether waiting a few milliseconds to serve several together pays "
         "for the delay it adds has not been re-measured on this engine."],
        ["Move the generator off the host", "A second machine, so the measurement stops "
         "competing with the thing it measures, and the two reserved cores go to the server."],
        ["The same campaign on x86", "The identical protocol on AMD and Intel hosts "
         "would tell a customer which processor to buy, and how many."],
        ["The other 39 languages", "This pack is multilingual and only English has been "
         "qualified."],
    ], [fw * 0.28, fw * 0.72], header=False))

    # ------------------------------------------------------------------ 12
    A(Paragraph("12. Reproducing this", H2))
    A(Paragraph(
        "The runtime, the benchmark harness and the qualified configuration are all "
        "source. The configuration is a committed file rather than a set of "
        "instructions, and the preflight refuses to run when the machine, the "
        "resolved kernels or the environment contradict it &mdash; so a repeat of "
        "this measurement cannot quietly become a measurement of something else.", BODY))
    A(Paragraph(
        "python3 tools/perf_profile.py check axion-c4a-highcpu32-nemotron-streaming \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--bin ./mynah-asr-server<br/>"
        "tools/bench/v2_qualify.sh -m models/nemotron-3.5-asr-streaming-0.6b \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--phase all --soak-c 80 --soak-seconds 1800 --soaks 2 \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--corpus samples/stress-en/manifest.json --corpus-sample 500 \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;-W 6 -T 5 -C 96 --http-threads 32 \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--server-cpus 0-29 --gen-cpus 30-31<br/>"
        "python3 tools/bench/v2_verdict.py &lt;run&gt;", MONO))
    A(Spacer(1, 6))
    A(Paragraph(
        "Measurements taken 22-23 September 2026 on GCP Axion c4a-highcpu-32. Model "
        "nvidia/nemotron-3.5-asr-streaming-0.6b, int8, preset [56, 3]. Corpus FLEURS "
        "English (CC-BY 4.0), bank 04a7753aa1e80f9a, 498 clips. Runtime mynah-asr, "
        "C11. Commit a969877. Prepared by Gabriele Mastrapasqua.", SMALL))

    doc.build(F)
    print("wrote", path)


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "asr-report.pdf")
