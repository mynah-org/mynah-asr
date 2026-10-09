import json, torch, soundfile as sf, numpy as np
from nemo.collections.asr.models import EncDecRNNTBPEModel
m = EncDecRNNTBPEModel.restore_from("runs/eou-it-5h-e52/final.nemo", map_location="cuda"); m.eval()
row = [json.loads(l) for l in open("runs/eou-it-5h-e52/train_5h_eou.json")][0]
a, sr = sf.read(row["audio_filepath"]); s=int(row.get("offset",0)*sr); a=a[s:s+int(row["duration"]*sr)]
a = np.concatenate([a, np.zeros(int(4*sr))])
x = torch.tensor(a, dtype=torch.float32, device="cuda")[None]
with torch.no_grad():
    f, fl = m.preprocessor(input_signal=x, length=torch.tensor([x.shape[1]], device="cuda"))
    e, el = m.encoder(audio_signal=f, length=fl)
def loss_for(text):
    ids = m.tokenizer.text_to_ids(text); y = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        d, dl, _ = m.decoder(targets=y, target_length=torch.tensor([len(ids)], device="cuda"))
        m.joint._fuse_loss_wer = False; j = m.joint(encoder_outputs=e, decoder_outputs=d)
        l = m.loss(log_probs=j, targets=y, input_lengths=el, target_lengths=torch.tensor([len(ids)], device="cuda"))
    return round(float(l),2), len(ids)
print("ROW loss text+EOU", loss_for(row["text"] + " <EOU>"))
print("ROW loss EOU only", loss_for("<EOU>"))
print("ROW loss text only", loss_for(row["text"]))
hy = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=e, encoded_lengths=el, return_hypotheses=True)
h = hy[0] if isinstance(hy,(list,tuple)) else hy; h = h[0] if isinstance(h,(list,tuple)) else h
print("ROW padded decode ids", (h.y_sequence.tolist() if hasattr(h.y_sequence,"tolist") else list(h.y_sequence))[:20])
