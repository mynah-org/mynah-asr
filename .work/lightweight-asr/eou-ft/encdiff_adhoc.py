import torch, json, soundfile as sf, numpy as np
from nemo.collections.asr.models import EncDecRNNTBPEModel
torch.set_num_threads(8)
s = EncDecRNNTBPEModel.restore_from("models/parakeet_realtime_eou_120m-v1.nemo", map_location="cpu").eval()
c = EncDecRNNTBPEModel.restore_from("diag/runs/eou-it-5h-e52/final.nemo", map_location="cpu").eval()
ss, cs = s.state_dict(), c.state_dict()
rel = []
for k in ss:
    if k.startswith("encoder.") and ss[k].dtype.is_floating_point and ss[k].numel() > 1000:
        rel.append((float((cs[k]-ss[k]).norm()/(ss[k].norm()+1e-9)), k))
rel.sort()
print("ROW encoder rel change: median %.3f max %.3f (%s)" % (rel[len(rel)//2][0], rel[-1][0], rel[-1][1]))
print("ROW first layers", [round(r,3) for r,k in rel if "layers.0." in k][:4], "last", [round(r,3) for r,k in rel if "layers.16." in k][:4])
r = json.loads(open("manifests/eval_fleurs_it.json").readline())
a, sr = sf.read(r["audio_filepath"]); x = torch.tensor(a, dtype=torch.float32)[None]
with torch.no_grad():
    outs = []
    for m in (s, c):
        f, fl = m.preprocessor(input_signal=x, length=torch.tensor([x.shape[1]]))
        e, el = m.encoder(audio_signal=f, length=fl); outs.append(e[0])
cos = torch.nn.functional.cosine_similarity(outs[0], outs[1], dim=0)
print("ROW encoder output cosine stock vs cold per frame: mean %.3f min %.3f; std stock %.3f cold %.3f; frame-to-frame var stock %.4f cold %.4f" % (
    float(cos.mean()), float(cos.min()), float(outs[0].std()), float(outs[1].std()),
    float(outs[0].std(dim=1).mean()), float(outs[1].std(dim=1).mean())))
