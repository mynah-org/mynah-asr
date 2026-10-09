#!/usr/bin/env python3
"""Coverage of a (stock) EOU tokenizer on a language's transcripts.

Reports tokens/word, <unk> share, round-trip failures and the characters that
map to <unk>. On 2026-10-09 the stock English SPE of the EOU 120M covered
Italian MLS/FLEURS text except accented vowels (-> <unk>), which is why
plain_asr.py de-accents its targets instead of swapping the tokenizer.

    $PY tok_coverage.py --nemo models/parakeet_realtime_eou_120m-v1.nemo \\
        --manifest manifests/train_5h.json --manifest manifests/eval_fleurs_it.json
"""
import argparse
import collections
import json
import re
import sys


def norm(t):  # what the stock model was trained on: lowercase, no punctuation except apostrophe
    t = t.lower()
    t = re.sub(r"[^\w' ]", " ", t)
    return " ".join(t.split())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--manifest", action="append", required=True)
    a = ap.parse_args()
    from nemo.collections.asr.models import EncDecRNNTBPEModel

    m = EncDecRNNTBPEModel.restore_from(a.nemo, map_location="cpu")
    tk = m.tokenizer
    sp = tk.tokenizer
    print("ROW vocab", tk.vocab_size, "unk id", sp.unk_id(), "pieces sample", [sp.id_to_piece(i) for i in range(0, 40, 4)])
    rows = []
    for mp in a.manifest:
        rows += [json.loads(line) for line in open(mp, encoding="utf-8") if line.strip()]
    unk = toks = words = rt_fail = 0
    bad = collections.Counter()
    for r in rows:
        t = norm(r["text"])
        ids = tk.text_to_ids(t)
        unk += sum(1 for i in ids if i == sp.unk_id())
        toks += len(ids)
        words += len(t.split())
        if tk.ids_to_text(ids) != t:
            rt_fail += 1
            for ch in t:
                if not tk.text_to_ids(ch) or sp.unk_id() in tk.text_to_ids(ch):
                    bad[ch] += 1
    print("ROW utts", len(rows), "words", words, "tokens/word %.2f" % (toks / max(1, words)), "unk tokens", unk,
          "(%.3f%% of tokens)" % (100 * unk / max(1, toks)), "round-trip failures", rt_fail)
    print("ROW chars that map to UNK:", bad.most_common(15))
    for r in rows[:4] + rows[-3:]:
        t = norm(r["text"])
        ids = tk.text_to_ids(t)
        print("ROW ex", repr(t[:70]), "->", [sp.id_to_piece(i) for i in ids[:14]], "| back:", repr(tk.ids_to_text(ids)[:70]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
