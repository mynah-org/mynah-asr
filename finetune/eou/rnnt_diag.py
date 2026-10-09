"""Small RNNT diagnostics shared by plain_asr.py and eou/diag/ (torch imported lazily).

Measured on 2026-10-09 (finetune/README.md): the collapsed EOU-IT runs had
max(non-blank) - blank < 0 on EVERY frame and beam search returned the same
sentence for every input. These helpers measure exactly that, cheaply:

- sos_blank_margin: per frame, max over non-blank logits minus the blank logit
  of the joint at the start-of-sequence prediction state. A lower bound on how
  close any token is to winning; all-negative = greedy emits nothing.
- nonblank_frac: share of frames where some token beats blank at that state.
- audio_dependence: distinct hypotheses / clips (1.0 = every clip different;
  ~1/N = an audio-independent text prior).
"""
from __future__ import annotations

import copy


def blank_id(model):
    return int(model.joint.num_classes_with_blank) - 1


def special_id(tokenizer, piece):
    """id of a special piece (<EOU>, <EOB>) or None when the tokenizer lacks it."""
    try:
        i = tokenizer.token_to_id(piece)
    except Exception:
        return None
    return i if isinstance(i, int) and i >= 0 and tokenizer.ids_to_text([i]).strip() == piece else None


def sos_blank_margin(model, enc, enc_len):
    """enc [B, D, T] encoder output -> list of {margin_max, margin_mean, nonblank_frac} per utterance."""
    import torch

    with torch.no_grad():
        d, _ = model.decoder.predict(None, state=None, add_sos=False, batch_size=enc.shape[0])[:2]
        j = model.joint.joint(enc.transpose(1, 2), d)[:, :, 0, :].float()
        blank = blank_id(model)
        mg = torch.cat([j[..., :blank], j[..., blank + 1:]], dim=-1).max(-1).values - j[..., blank]
    out = []
    for b in range(enc.shape[0]):
        t = int(enc_len[b])
        x = mg[b, :t]
        out.append({"margin_max": float(x.max()), "margin_mean": float(x.mean()),
                    "nonblank_frac": float((x > 0).float().mean())})
    return out


def hyp_ids(h):
    h = h[0] if isinstance(h, (list, tuple)) else h
    ys = h.y_sequence
    return ys.tolist() if hasattr(ys, "tolist") else list(ys)


def set_strategy(model, strategy, beam_size=4):
    """Switch decoding to 'greedy_batch' or 'beam' (beam_size, best hypothesis);
    returns the previous decoding cfg so the caller can restore it."""
    from omegaconf import open_dict

    prev = copy.deepcopy(model.cfg.decoding)
    cfg = copy.deepcopy(model.cfg.decoding)
    with open_dict(cfg):
        cfg.strategy = strategy
        if strategy == "beam":
            cfg.beam.beam_size = beam_size
            cfg.beam.return_best_hypothesis = True
    model.change_decoding_strategy(cfg, verbose=False) if _accepts_verbose(model) else model.change_decoding_strategy(cfg)
    return prev


def restore_strategy(model, prev):
    model.change_decoding_strategy(prev, verbose=False) if _accepts_verbose(model) else model.change_decoding_strategy(prev)


def _accepts_verbose(model):
    import inspect

    try:
        return "verbose" in inspect.signature(model.change_decoding_strategy).parameters
    except (TypeError, ValueError):
        return False


def audio_dependence(hyps):
    hyps = [h.strip() for h in hyps]
    return round(len(set(hyps)) / max(1, len(hyps)), 4)
