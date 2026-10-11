"""Token-level (SPE id) interpolated Kneser-Ney ARPA for NeMo's NGPU-LM, pure Python, no KenLM.

NeMo 3.0 fuses an n-gram LM into RNNT greedy_batch / malsd_batch decoding
(`decoding.greedy.ngram_lm_model` / `decoding.beam.ngram_lm_model`, loaded by
NGramGPULanguageModel.from_file). The ARPA must be over SPE token ids written as
chr(id + 100) (DEFAULT_TOKEN_OFFSET). The usual builder (scripts/asr_language_modeling/
ngram_lm/train_kenlm.py) needs the KenLM binaries, which are not installed on the box;
this writes the same format directly.

  CUDA_VISIBLE_DEVICES= python build_token_arpa.py --nemo /root/ft/models/parakeet_realtime_eou_120m-v1.nemo \
      --train a.json,b.json[,corpus.txt] --exclude val1.json,val2.json --order 4 --out lm4.arpa \
      --eval mls=/root/ft/manifests/eval_mls_it.json,...

Text files (.txt, one sentence per line) and NeMo manifests (.json, "text" field) are both
accepted; text is normalised like two_stage.py (lower case, de-accented, [a-z' ]).
Sentences that appear (normalised) in --exclude are dropped from training. With --eval it
also prints token- and word-level perplexity of the LM on those sets (no </s>, to compare
with pred_ilm_probe.py's ILM perplexity).
"""
import argparse, io, json, math, re, tarfile, unicodedata
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--nemo", required=True, help=".nemo whose tokenizer.model defines the ids")
ap.add_argument("--train", required=True)
ap.add_argument("--exclude", default="")
ap.add_argument("--order", type=int, default=4)
ap.add_argument("--offset", type=int, default=100)
ap.add_argument("--vocab", type=int, default=1026, help="LM vocabulary = blank id (pieces + <EOU>/<EOB>)")
ap.add_argument("--out", required=True)
ap.add_argument("--eval", default="")
a = ap.parse_args()


def norm(t):
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z' ]", " ", t)
    return " ".join(t.split())


def read(path):
    if path.endswith(".json"):
        return [norm(json.loads(l)["text"]) for l in open(path)]
    return [norm(l) for l in open(path, errors="ignore")]


t = tarfile.open(a.nemo)
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_proto=t.extractfile([n for n in t.getnames() if n.endswith("tokenizer.model")][0]).read())

excl = {s for p in a.exclude.split(",") if p for s in read(p)}
sents, dropped = [], 0
for p in a.train.split(","):
    for s in read(p):
        if not s:
            continue
        if s in excl:
            dropped += 1; continue
        sents.append(s)
print(f"train sentences {len(sents)} (dropped {dropped} also in --exclude)", flush=True)

BOS, EOS = "<s>", "</s>"
N = a.order
raw = [Counter() for _ in range(N + 1)]
ntok = 0
for s in sents:
    ids = [BOS] + sp.encode(s) + [EOS]
    ntok += len(ids) - 2
    for k in range(1, N + 1):
        for i in range(len(ids) - k + 1):
            raw[k][tuple(ids[i:i + k])] += 1
print(f"tokens {ntok}, ngrams per order {[len(c) for c in raw[1:]]}", flush=True)

# adjusted counts: raw for the top order and for n-grams starting with <s>, else continuation counts
adj = [None] + [Counter() for _ in range(N)]
adj[N] = raw[N]
for k in range(N - 1, 0, -1):
    cont = Counter()
    for g in raw[k + 1]:
        cont[g[1:]] += 1
    for g, c in raw[k].items():
        adj[k][g] = c if g[0] == BOS else cont.get(g, 0)
    for g in list(adj[k]):
        if adj[k][g] == 0:
            del adj[k][g]      # only possible for <s>-less contexts never extended; kept by raw if needed
        # (every k-gram is a suffix of some (k+1)-gram except sentence-initial ones, handled above)


def discount(c):
    n1 = sum(1 for v in c.values() if v == 1); n2 = sum(1 for v in c.values() if v == 2)
    return n1 / (n1 + 2 * n2) if n1 + 2 * n2 else 0.5


D = [0.0] + [discount(adj[k]) for k in range(1, N + 1)]
prob = [None] + [dict() for _ in range(N)]
bow = [None] + [dict() for _ in range(N)]
# unigrams (interpolated with uniform over the vocabulary + </s>)
tot1 = sum(c for g, c in adj[1].items() if g[0] != BOS)
types1 = sum(1 for g in adj[1] if g[0] != BOS)
unif = 1.0 / (a.vocab + 1)
g1 = D[1] * types1 / tot1
for g, c in adj[1].items():
    if g[0] == BOS:
        continue
    prob[1][g] = max(c - D[1], 0) / tot1 + g1 * unif
prob[1][("<unk>",)] = g1 * unif
for k in range(2, N + 1):
    den, nfol = defaultdict(int), defaultdict(int)
    for g, c in adj[k].items():
        den[g[:-1]] += c; nfol[g[:-1]] += 1
    for h in den:
        bow[k - 1][h] = D[k] * nfol[h] / den[h]
    for g, c in adj[k].items():
        h = g[:-1]
        prob[k][g] = max(c - D[k], 0) / den[h] + bow[k - 1][h] * prob[k - 1][g[1:]]


def sym(x):
    return x if isinstance(x, str) else chr(x + a.offset)


with open(a.out, "w") as f:
    f.write("\\data\\\n")
    f.write(f"ngram 1={len(prob[1]) + 1}\n")
    for k in range(2, N + 1):
        f.write(f"ngram {k}={len(prob[k])}\n")
    for k in range(1, N + 1):
        f.write(f"\n\\{k}-grams:\n")
        if k == 1:
            f.write(f"-99\t{BOS}\t{math.log10(bow[1].get((BOS,), 1.0)):.6f}\n")
        for g, p in prob[k].items():
            s = " ".join(sym(x) for x in g)
            if k < N and g in bow[k]:
                f.write(f"{math.log10(p):.6f}\t{s}\t{math.log10(bow[k][g]):.6f}\n")
            else:
                f.write(f"{math.log10(p):.6f}\t{s}\n")
    f.write("\n\\end\\\n")
print(f"wrote {a.out} (order {N}, discounts {[round(x, 3) for x in D[1:]]})", flush=True)


def lp(hist, w):
    """natural-log backoff probability of w after the token tuple hist."""
    acc = 0.0
    for k in range(min(N - 1, len(hist)), -1, -1):
        h = tuple(hist[len(hist) - k:]) if k else ()
        g = h + (w,)
        if g in prob[k + 1]:
            return acc + math.log(prob[k + 1][g])
        if k and h in bow[k]:
            acc += math.log(bow[k][h])
    return acc + math.log(prob[1][("<unk>",)])


for spec in [x for x in a.eval.split(",") if x]:
    n, p = spec.split("=", 1)
    ss = [s for s in read(p) if s]
    nll, nt, nw = 0.0, 0, 0
    for s in ss:
        ids = sp.encode(s); hist = [BOS]
        for w in ids:
            nll -= lp(hist, w); hist.append(w); nt += 1
        nw += len(s.split())
    print(f"EVAL {n}: sents {len(ss)} tok/word {nt / nw:.2f} ppl_tok {math.exp(nll / nt):.2f} ppl_word {math.exp(nll / nw):.1f}", flush=True)
