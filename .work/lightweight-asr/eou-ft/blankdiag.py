"""Cold EOU-IT checkpoint: blank dominance vs beam search, on CPU (the GPU is training).
Per clip: greedy text, beam-4 text, and per-frame margin max(non-blank) - blank of the
joint at the SOS prediction state (a lower bound on how close a token is to winning)."""
import json, sys, torch, numpy as np, soundfile as sf
from huggingface_hub import hf_hub_download
from nemo.collections.asr.models import EncDecRNNTBPEModel
from omegaconf import open_dict
torch.set_num_threads(8)
p = hf_hub_download("gabrione/mynah-asr-canary-it-experiments", "runs/eou-it-5h-e52/final.nemo",
                    token=open("/root/.hf_token").read().strip(), local_dir="/root/ft/diag")
m = EncDecRNNTBPEModel.restore_from(p, map_location="cpu"); m.eval()
m.joint._fuse_loss_wer = False
tr = [json.loads(l) for l in open("/root/ft/runs/eou-it-5h-e52/train_5h_eou.json")][:5]
ev = [json.loads(l) for l in open("/root/ft/manifests/eval_fleurs_it.json")][:5]
def enc(r, pad_s=3.0):
    a, sr = sf.read(r["audio_filepath"]); s = int(r.get("offset", 0) * sr)
    a = a[s:s + int(r["duration"] * sr)] if r.get("offset") is not None else a
    a = np.concatenate([a, np.zeros(int(pad_s * sr))])
    x = torch.tensor(a, dtype=torch.float32)[None]
    with torch.no_grad():
        f, fl = m.preprocessor(input_signal=x, length=torch.tensor([x.shape[1]]))
        return m.encoder(audio_signal=f, length=fl)
def text_of(e, el):
    with torch.no_grad():
        hy = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=e, encoded_lengths=el, return_hypotheses=True)
    h = hy[0] if isinstance(hy, (list, tuple)) else hy
    h = h[0] if isinstance(h, (list, tuple)) else h
    ys = h.y_sequence.tolist() if hasattr(h.y_sequence, "tolist") else list(h.y_sequence)
    return m.tokenizer.ids_to_text([i for i in ys if i < 1024]), ys.count(1024)
res = {}
for strat in ("greedy_batch", "beam"):
    cfg = m.cfg.decoding
    with open_dict(cfg):
        cfg.strategy = strat
        if strat == "beam":
            cfg.beam.beam_size = 4; cfg.beam.return_best_hypothesis = True
    m.change_decoding_strategy(cfg)
    for name, rows in (("train", tr), ("eval", ev)):
        for i, r in enumerate(rows):
            e, el = enc(r)
            t, n_eou = text_of(e, el)
            res.setdefault((name, i), {"ref": r["text"][:60]})[strat] = (t[:60], n_eou)
            if strat == "greedy_batch":
                with torch.no_grad():
                    d, _ = m.decoder.predict(None, state=None, add_sos=False, batch_size=1)[:2]
                    j = m.joint.joint(e.transpose(1, 2), d)[0, :, 0, :]
                margin = (j[:, :1026].max(-1).values - j[:, 1026])
                res[(name, i)]["margin"] = [round(float(margin.max()), 2), round(float(margin.mean()), 2),
                                            int((margin > 0).sum()), int(margin.numel())]
for k, v in res.items():
    print("ROW", k, "| ref:", v["ref"], "| greedy:", v["greedy_batch"], "| beam4:", v["beam"],
          "| margin max/mean/frames>0/frames:", v["margin"])
