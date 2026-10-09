"""Stock EOU-120M tokenizer coverage on Italian transcripts (MLS-it train 5h + FLEURS-it test)."""
import json, collections
from nemo.collections.asr.models import EncDecRNNTBPEModel
m = EncDecRNNTBPEModel.restore_from("/root/ft/models/parakeet_realtime_eou_120m-v1.nemo", map_location="cpu")
tk = m.tokenizer; sp = tk.tokenizer
print("ROW vocab", tk.vocab_size, "unk id", sp.unk_id(), "pieces sample", [sp.id_to_piece(i) for i in range(0, 40, 4)])
rows = [json.loads(l) for l in open("/root/ft/manifests/train_5h.json")] + [json.loads(l) for l in open("/root/ft/manifests/eval_fleurs_it.json")]
def norm(t):  # what the stock model was trained on: lowercase, no punctuation except apostrophe
    import re
    t = t.lower()
    t = re.sub(r"[^\w' ]", " ", t)
    return " ".join(t.split())
unk = toks = words = 0; bad_chars = collections.Counter(); rt_fail = 0
for r in rows:
    t = norm(r["text"]); ids = tk.text_to_ids(t)
    unk += sum(1 for i in ids if i == sp.unk_id()); toks += len(ids); words += len(t.split())
    if tk.ids_to_text(ids) != t:
        rt_fail += 1
        for ch in t:
            if not tk.text_to_ids(ch) or sp.unk_id() in tk.text_to_ids(ch):
                bad_chars[ch] += 1
print("ROW utts", len(rows), "words", words, "tokens/word %.2f" % (toks / words), "unk tokens", unk,
      "(%.3f%% of tokens)" % (100 * unk / toks), "round-trip failures", rt_fail)
print("ROW chars that map to UNK:", bad_chars.most_common(15))
for r in rows[:4] + rows[-3:]:
    t = norm(r["text"]); ids = tk.text_to_ids(t)
    print("ROW ex", repr(t[:70]), "->", [sp.id_to_piece(i) for i in ids[:14]], "| back:", repr(tk.ids_to_text(ids)[:70]))
