#!/usr/bin/env python3
"""Convert an HF-native NeMo ASR checkpoint into the Mynah layout (in-place).

Input: directory holding the HuggingFace port of the model
  (config.json, processor_config.json, tokenizer.json, model.safetensors).
Output (written INTO the same directory; the weights are left untouched):
  - mynah.json              unified config that drives the C runtime
  - tokens.json             id -> piece array (blank included, last)
  - mel_filters.safetensors precomputed mel filterbank (slaney) + Hann window

Supported models (dispatched on config.json model_type):
  - nemotron3_5_asr  (cache-aware streaming, RNNT, language prompt)
  - parakeet_tdt     (offline, TDT, no prompt — docs/parakeet-tdt-arch.md)

Alternative input: a directory holding ONLY the `.nemo` archive (models without
an HF-native port, e.g. parakeet-tdt_ctc-110m). Requires torch
(`uv sync --extra oracle`): it extracts model_config.yaml + tokenizer, renames
the NeMo tensors to the canonical (HF) naming the runtime reads, and writes
model.safetensors in f32.

Usage: uv run python convert_nemo.py ../models/<model_dir>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


# ---------------------------------------------------------------- mel filters
def hz_to_mel_slaney(f: np.ndarray) -> np.ndarray:
    """Scala mel 'slaney' (librosa/torchaudio htk=False): lineare sotto 1 kHz."""
    f = np.asarray(f, dtype=np.float64)
    mel = 3.0 * f / 200.0
    log_region = f >= 1000.0
    mel = np.where(log_region, 15.0 + 27.0 * np.log(np.maximum(f, 1e-10) / 1000.0) / np.log(6.4), mel)
    return mel


def mel_to_hz_slaney(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    f = 200.0 * m / 3.0
    log_region = m >= 15.0
    f = np.where(log_region, 1000.0 * np.exp(np.log(6.4) * (m - 15.0) / 27.0), f)
    return f


def melscale_fbanks(n_freqs: int, f_min: float, f_max: float, n_mels: int, sample_rate: int) -> np.ndarray:
    """Clone di torchaudio.functional.melscale_fbanks(norm='slaney', mel_scale='slaney').

    Identica alla costruzione NeMo (librosa htk=False, norm slaney); vedi
    docs/prior-art.md §B.2 (onnx-asr la valida vs NeMo con atol 5e-7).
    Ritorna [n_freqs, n_mels] float64.
    """
    all_freqs = np.linspace(0.0, sample_rate // 2, n_freqs, dtype=np.float64)
    m_pts = mel_to_hz_slaney(np.linspace(hz_to_mel_slaney(f_min), hz_to_mel_slaney(f_max), n_mels + 2))
    f_diff = m_pts[1:] - m_pts[:-1]                        # [n_mels+1]
    slopes = m_pts[None, :] - all_freqs[:, None]           # [n_freqs, n_mels+2]
    down = -slopes[:, :-2] / f_diff[:-1]                   # [n_freqs, n_mels]
    up = slopes[:, 2:] / f_diff[1:]
    fb = np.maximum(0.0, np.minimum(down, up))
    fb *= 2.0 / (m_pts[2:] - m_pts[:-2])                   # norm slaney
    return fb


# ------------------------------------------------------------------ tokenizer
def load_pieces(model_dir: Path, vocab_size: int, blank_id: int, blank_token: str) -> list[str]:
    """Estrae l'array id -> piece dal tokenizer.json (formato HF tokenizers).

    Nota (verificato sul checkpoint): il tokenizer.json elenca <pad>=13087 e <blank>=13088,
    ma nello spazio di output del modello (joint = vocab_size=13088 logit, blank_as_pad)
    l'id 13087 È il blank/pad e 13088 non esiste. Normalizziamo: pieces[blank_id]=blank_token,
    gli added token oltre vocab_size vengono ignorati.
    """
    tok = json.loads((model_dir / "tokenizer.json").read_text())
    pieces: list[str | None] = [None] * vocab_size
    for piece, idx in tok["model"]["vocab"].items():
        pieces[idx] = piece
    for added in tok.get("added_tokens", []):
        if added["id"] < blank_id:
            pieces[added["id"]] = added["content"]
    pieces[blank_id] = blank_token
    missing = [i for i, p in enumerate(pieces) if p is None]
    assert not missing, f"{len(missing)} id senza piece (primi: {missing[:5]})"
    return pieces  # type: ignore[return-value]


# ----------------------------------------------------------------------- main
def features_section(fe: dict, normalize: str) -> dict:
    return {
        "sample_rate": fe["sampling_rate"],
        "n_mels": fe["feature_size"],
        "n_fft": fe["n_fft"],
        "win_length": fe["win_length"],
        "hop_length": fe["hop_length"],
        "preemphasis": fe["preemphasis"],
        "log_zero_guard": 2.0 ** -24,
        "normalize": normalize,
        "dither": 0.0,                      # inference
        "mel_filters": "mel_filters.safetensors",
    }


def build_nemotron(model_dir: Path, cfg: dict, proc: dict) -> dict:
    enc = cfg["encoder_config"]
    fe = proc["feature_extractor"]
    sub = enc["subsampling_factor"]
    left_context = enc["sliding_window"] - 1
    lookaheads = enc["supported_num_lookahead_tokens"]
    frame_ms = fe["hop_length"] / fe["sampling_rate"] * sub * 1000.0

    return {
        "mynah_format": 1,
        "name": model_dir.name,
        "arch": "fastconformer_rnnt_streaming",
        "engine": "nemotron-streaming",
        "weights": "model.safetensors",
        # normalize "NA" dal model_config.yaml nel .nemo (vedi docs/nemotron-arch.md)
        "features": features_section(fe, "NA"),
        "encoder": {
            "n_layers": enc["num_hidden_layers"],
            "d_model": enc["hidden_size"],
            "n_heads": enc["num_attention_heads"],
            "ffn_dim": enc["intermediate_size"],
            "conv_kernel": enc["conv_kernel_size"],
            "conv_norm": "layer_norm",
            "subsampling": "dw_striding_causal",
            "subsampling_factor": sub,
            "subsampling_conv_channels": enc["subsampling_conv_channels"],
            "use_bias": enc["attention_bias"],
            "activation": enc["hidden_act"],          # silu
            "att_context_style": "chunked_limited",
            "pos_emb_max_len": enc["max_position_embeddings"],
            "xscaling": False,
        },
        "decoder": {
            "type": "rnnt_lstm",
            "pred_hidden": cfg["decoder_hidden_size"],
            "pred_layers": cfg["num_decoder_layers"],
            "joint_hidden": cfg["decoder_hidden_size"],
            "joint_activation": cfg["hidden_act"],    # relu
            "vocab_size": cfg["vocab_size"],
            "blank_id": cfg["blank_token_id"],
            "max_symbols_per_step": cfg["max_symbols_per_step"],
        },
        "streaming": {
            "att_context_presets": [[left_context, r] for r in lookaheads],
            "default_preset_index": lookaheads.index(enc["default_num_lookahead_tokens"]),
            "encoder_frame_ms": frame_ms,
        },
        "prompt": {
            "num_prompts": cfg["num_prompts"],
            "default_id": cfg["default_prompt_id"],
            "intermediate_size": cfg["prompt_intermediate_size"],
            "dictionary": proc["prompt_dictionary"],
        },
        "tokenizer": {"type": "spe_bpe", "pieces": "tokens.json"},
    }


def build_parakeet_tdt(model_dir: Path, cfg: dict, proc: dict) -> dict:
    """parakeet-tdt (offline, non-causale, conv batch_norm, decoder TDT).
    Riferimento numerico: docs/parakeet-tdt-arch.md + reference/transformers-parakeet/."""
    enc = cfg["encoder_config"]
    fe = proc["feature_extractor"]

    return {
        "mynah_format": 1,
        "name": model_dir.name,
        "arch": "fastconformer_tdt",
        "engine": "parakeet-tdt",
        "weights": "model.safetensors",
        # normalize per_feature dal model_config.yaml nel .nemo (parakeet-tdt-arch.md)
        "features": features_section(fe, "per_feature"),
        "encoder": {
            "n_layers": enc["num_hidden_layers"],
            "d_model": enc["hidden_size"],
            "n_heads": enc["num_attention_heads"],
            "ffn_dim": enc["intermediate_size"],
            "conv_kernel": enc["conv_kernel_size"],
            "conv_norm": "batch_norm",                # foldata in scale+shift al load
            "subsampling": "dw_striding",             # NON causale: padding simmetrico
            "subsampling_factor": enc["subsampling_factor"],
            "subsampling_conv_channels": enc["subsampling_conv_channels"],
            "use_bias": enc["attention_bias"],
            "activation": enc["hidden_act"],          # silu
            "att_context_style": "regular",           # attention full [-1, -1]
            "pos_emb_max_len": enc["max_position_embeddings"],
            "xscaling": bool(enc.get("scale_input", False)),
        },
        "decoder": {
            "type": "tdt_lstm",
            "pred_hidden": cfg["decoder_hidden_size"],
            "pred_layers": cfg["num_decoder_layers"],
            "joint_hidden": cfg["decoder_hidden_size"],
            "joint_activation": cfg["hidden_act"],    # relu
            "vocab_size": cfg["vocab_size"],          # 8193 = 8192 pieces + blank
            "blank_id": cfg["blank_token_id"],        # 8192
            "max_symbols_per_step": cfg["max_symbols_per_step"],
            "durations": cfg["durations"],            # [0, 1, 2, 3, 4]
        },
        # niente sezione "streaming" (modello offline) ne' "prompt" (LID implicita)
        "tokenizer": {"type": "spe_bpe", "pieces": "tokens.json"},
    }


BUILDERS = {
    "nemotron3_5_asr": build_nemotron,
    "parakeet_tdt": build_parakeet_tdt,
}


# ------------------------------------------------------------- ingresso .nemo
# Rename NeMo -> canonico (naming del porting HF, quello che il runtime legge).
# Verificato su parakeet-tdt_ctc-110m (docs/parakeet-tdt-arch.md per la mappa v3).
_NEMO_RENAMES = [
    ("encoder.pre_encode.conv.", "encoder.subsampling.layers."),
    ("encoder.pre_encode.out.", "encoder.subsampling.linear."),
    (".self_attn.linear_q.", ".self_attn.q_proj."),
    (".self_attn.linear_k.", ".self_attn.k_proj."),
    (".self_attn.linear_v.", ".self_attn.v_proj."),
    (".self_attn.linear_out.", ".self_attn.o_proj."),
    (".self_attn.linear_pos.", ".self_attn.relative_k_proj."),
    (".self_attn.pos_bias_u", ".self_attn.bias_u"),
    (".self_attn.pos_bias_v", ".self_attn.bias_v"),
    (".conv.batch_norm.", ".conv.norm."),
    ("decoder.prediction.embed.", "decoder.embedding."),
    ("decoder.prediction.dec_rnn.lstm.", "decoder.lstm."),
    ("joint.enc.", "encoder_projector."),
    ("joint.pred.", "decoder.decoder_projector."),
]

# Rename del ramo AED (EncDecMultiTaskModel: Canary) — docs/canary-arch.md.
_AED_RENAMES = [
    ("encoder_decoder_proj.", "enc_dec_proj."),
    ("transf_decoder._embedding.token_embedding.weight", "aed.embedding.weight"),
    ("transf_decoder._embedding.position_embedding.pos_enc", "aed.pos_enc"),
    ("transf_decoder._embedding.layer_norm.", "aed.emb_norm."),
    ("transf_decoder._decoder.final_layer_norm.", "aed.final_norm."),
    ("transf_decoder._decoder.layers.", "aed.layers."),
    ("log_softmax.mlp.layer0.", "aed.head."),
    (".layer_norm_1.", ".ln_self."),
    (".first_sub_layer.", ".self_attn."),
    (".layer_norm_2.", ".ln_cross."),
    (".second_sub_layer.", ".cross_attn."),
    (".layer_norm_3.", ".ln_ffn."),
    (".third_sub_layer.", ".ffn."),
    (".query_net.", ".q_proj."),
    (".key_net.", ".k_proj."),
    (".value_net.", ".v_proj."),
    (".out_projection.", ".o_proj."),
    (".dense_in.", ".linear1."),
    (".dense_out.", ".linear2."),
]


def rename_nemo_key(k: str) -> str | None:
    """None = tensore da saltare (preprocessor, stats inutili, head CTC ausiliaria)."""
    if k.startswith("preprocessor.") or k.endswith("num_batches_tracked"):
        return None
    if k.startswith(("transf_decoder.", "log_softmax.", "encoder_decoder_proj.")):
        for old, new in _AED_RENAMES:
            k = k.replace(old, new)
        return k
    m = re.fullmatch(r"joint\.joint_net\.\d+\.(weight|bias)", k)
    if m:
        return f"joint.head.{m.group(1)}"
    # head CTC: ausiliaria dei modelli hybrid (ctc_decoder.*) o decoder principale
    # dei modelli CTC puri (decoder.decoder_layers.*)
    m = re.fullmatch(r"(?:ctc_)?decoder\.decoder_layers\.0\.(weight|bias)", k)
    if m:
        return f"ctc_head.{m.group(1)}"
    for old, new in _NEMO_RENAMES:
        k = k.replace(old, new)
    return k


def yaml_features_section(fe: dict) -> dict:
    sr = fe["sample_rate"]
    return {
        "sample_rate": sr,
        "n_mels": fe["features"],
        "n_fft": fe["n_fft"],
        "win_length": round(fe["window_size"] * sr),
        "hop_length": round(fe["window_stride"] * sr),
        "preemphasis": 0.97,                 # default NeMo (assente nello yaml)
        "log_zero_guard": 2.0 ** -24,
        "normalize": fe["normalize"],
        "dither": 0.0,
        "mel_filters": "mel_filters.safetensors",
    }


def yaml_encoder_section(enc: dict) -> dict:
    assert enc.get("att_context_style", "regular") == "regular" \
        and not enc.get("causal_downsampling", False), \
        "solo encoder offline non-causali da .nemo (per ora)"
    return {
        "n_layers": enc["n_layers"],
        "d_model": enc["d_model"],
        "n_heads": enc["n_heads"],
        "ffn_dim": enc["d_model"] * enc["ff_expansion_factor"],
        "conv_kernel": enc["conv_kernel_size"],
        "conv_norm": enc["conv_norm_type"],
        "subsampling": enc["subsampling"],            # dw_striding (non causale)
        "subsampling_factor": enc["subsampling_factor"],
        "subsampling_conv_channels": enc["subsampling_conv_channels"],
        "use_bias": enc.get("use_bias", True),        # default NeMo: bias presenti
        "activation": "silu",
        "att_context_style": enc.get("att_context_style", "regular"),
        "pos_emb_max_len": enc["pos_emb_max_len"],
        # xscaling: input dei layer scalato per sqrt(d_model) (modelli piu' vecchi)
        "xscaling": bool(enc.get("xscaling", False)),
    }


def build_parakeet_tdt_from_yaml(model_dir: Path, y: dict) -> dict:
    """mynah.json dal model_config.yaml del .nemo — Parakeet RNNT/TDT (anche hybrid)."""
    dec = y["decoder"]
    durations = y["decoding"].get("durations") or []
    return {
        "mynah_format": 1,
        "name": model_dir.name,
        "arch": "fastconformer_tdt" if durations else "fastconformer_rnnt",
        "engine": "parakeet-tdt" if durations else "parakeet-rnnt",
        "weights": "model.safetensors",
        "features": yaml_features_section(y["preprocessor"]),
        "encoder": yaml_encoder_section(y["encoder"]),
        "decoder": {
            "type": "tdt_lstm" if durations else "rnnt_lstm",
            "pred_hidden": dec["prednet"]["pred_hidden"],
            "pred_layers": dec["prednet"]["pred_rnn_layers"],
            "joint_hidden": y["joint"]["jointnet"]["joint_hidden"],
            "joint_activation": y["joint"]["jointnet"]["activation"],
            "vocab_size": y["joint"]["num_classes"] + 1,  # + blank
            "blank_id": y["joint"]["num_classes"],
            "max_symbols_per_step": y["decoding"]["greedy"]["max_symbols"],
            **({"durations": durations} if durations else {}),
        },
        "tokenizer": {"type": "spe_bpe", "pieces": "tokens.json"},
    }


def build_parakeet_ctc_from_yaml(model_dir: Path, y: dict) -> dict:
    """Parakeet CTC puro (ConvASRDecoder): niente prednet/joint, solo ctc_head."""
    return {
        "mynah_format": 1,
        "name": model_dir.name,
        "arch": "fastconformer_ctc",
        "engine": "parakeet-ctc",
        "weights": "model.safetensors",
        "features": yaml_features_section(y["preprocessor"]),
        "encoder": yaml_encoder_section(y["encoder"]),
        "decoder": {
            "type": "ctc",
            "vocab_size": y["decoder"]["num_classes"] + 1,  # + blank (ultimo indice)
            "blank_id": y["decoder"]["num_classes"],
            "max_symbols_per_step": 1,                      # n/a per CTC
        },
        "tokenizer": {"type": "spe_bpe", "pieces": "tokens.json"},
    }


# canary-1b-v2 (tokenizer unificato): lingue dalla model card, validate contro
# il vocab alla conversione (il token <|xx|> deve esistere)
CANARY_V2_LANGS = ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr",
                   "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro",
                   "ru", "sk", "sl", "sv", "uk"]


def build_canary_from_yaml(model_dir: Path, y: dict) -> dict:
    """Canary Flash/v2 (EncDecMultiTaskModel): AED con decoder Transformer.
    docs/canary-arch.md. I token del prompt restano STRINGHE nel mynah.json:
    il runtime li risolve in id via tokens.json (niente doppia contabilità)."""
    dcfg = y["transf_decoder"]["config_dict"]
    # aggregato (flash): lingue = chiavi dei sub-tokenizer; unificato (v2): model card
    langs = ([l for l in y["tokenizer"]["langs"] if l != "spl_tokens"]
             if y["tokenizer"].get("type") == "agg" else CANARY_V2_LANGS)
    slots = y["prompt_defaults"][0]["slots"]
    assert y["prompt_format"] == "canary2", f"prompt_format non supportato: {y['prompt_format']}"
    assert dcfg["pre_ln"] and dcfg["hidden_act"] == "relu"
    return {
        "mynah_format": 1,
        "name": model_dir.name,
        "arch": "fastconformer_aed",
        "engine": "canary-aed",
        "weights": "model.safetensors",
        "features": yaml_features_section(y["preprocessor"]),
        "encoder": yaml_encoder_section(y["encoder"]),
        "decoder": {
            "type": "aed_transformer",
            "n_layers": dcfg["num_layers"],
            "d_model": dcfg["hidden_size"],
            "n_heads": dcfg["num_attention_heads"],
            "ffn_dim": dcfg["inner_size"],
            "max_seq": dcfg["max_sequence_length"],
            "vocab_size": y["head"]["num_classes"],
            "eos_token": "<|endoftext|>",
            "max_generation_delta": y["decoding"]["beam"].get("max_generation_delta", 50),
        },
        "prompt": {
            "format": "canary2",
            # template canary2 (Canary2PromptFormatter): boctx, contesto (vuoto di
            # default), bos, poi gli slot in quest'ordine
            "template": ["<|startofcontext|>", "{decodercontext}", "<|startoftranscript|>",
                         "{emotion}", "{source_lang}", "{target_lang}", "{pnc}",
                         "{itn}", "{timestamp}", "{diarize}"],
            "defaults": {k: v for k, v in slots.items()},
            "languages": langs,          # <|xx|> come token; xx = codice lingua
        },
        "tokenizer": {"type": "spe_bpe_agg", "pieces": "tokens.json"},
    }


def canary_pieces(tar, names, y: dict) -> tuple[list[str], int]:
    """Tokenizer Canary: aggregato (flash — pezzi in ordine di id GLOBALE,
    sub-tokenizer nell'ordine dello yaml, offset cumulativi) o unificato (v2 —
    un solo modello SPE). Ritorna (pieces, spl_size; 0 = unificato)."""
    import sentencepiece as spm

    def load(member: str) -> "spm.SentencePieceProcessor":
        hits = [n for n in names if n.endswith(member.replace("nemo:", ""))]
        assert len(hits) == 1, f"{member}: {hits}"
        sp = spm.SentencePieceProcessor()
        sp.load_from_serialized_proto(tar.extractfile(hits[0]).read())
        return sp

    if y["tokenizer"].get("type") != "agg":
        sp = load(y["tokenizer"]["model_path"])
        return [sp.id_to_piece(i) for i in range(sp.get_piece_size())], 0

    pieces: list[str] = []
    spl_size = 0
    for lang, sub in y["tokenizer"]["langs"].items():
        sp = load(sub["model_path"])
        n = sp.get_piece_size()
        pieces += [sp.id_to_piece(i) for i in range(n)]
        if lang == "spl_tokens":
            spl_size = n
    return pieces, spl_size


def convert_from_nemo(model_dir: Path, nemo_file: Path) -> None:
    import io
    import tarfile

    import torch  # extra 'oracle': uv sync --extra oracle
    import yaml

    tar = tarfile.open(nemo_file)
    names = tar.getnames()

    def member(suffix: str) -> str:
        hits = [n for n in names if n.endswith(suffix)]
        if len(hits) > 1:   # es. v2: model_config.yaml + timestamps_asr_model_config.yaml
            exact = [n for n in hits if n.rsplit("/", 1)[-1] == suffix]
            if exact:
                hits = exact
        assert len(hits) == 1, f"{suffix}: {hits}"
        return hits[0]

    ycfg = yaml.safe_load(tar.extractfile(member("model_config.yaml")).read())
    aed = ycfg.get("target", "").endswith("EncDecMultiTaskModel")
    if aed:
        mynah = build_canary_from_yaml(model_dir, ycfg)
        # v2 bundla un ALLINEATORE separato per i timestamp: il token
        # <|timestamp|> generativo non funziona (il modello emette EOS subito)
        if any(n.endswith("timestamps_asr_model_config.yaml") for n in names):
            mynah["decoder"]["timestamp_tokens"] = False
    elif "ConvASRDecoder" in ycfg["decoder"]["_target_"]:
        mynah = build_parakeet_ctc_from_yaml(model_dir, ycfg)
    else:
        mynah = build_parakeet_tdt_from_yaml(model_dir, ycfg)

    sd = torch.load(io.BytesIO(tar.extractfile(member("model_weights.ckpt")).read()),
                    map_location="cpu", weights_only=True)

    tensors: dict[str, np.ndarray] = {}
    for k, v in sd.items():
        nk = rename_nemo_key(k)
        if nk is None:
            continue
        assert nk not in tensors, f"rename duplicato: {k} -> {nk}"
        tensors[nk] = v.float().numpy()
    save_file(tensors, model_dir / "model.safetensors")

    # filterbank e finestra DAL checkpoint (bit-esatte con NeMo)
    fb = sd["preprocessor.featurizer.fb"].numpy()[0].T          # [1,M,257] -> [257,M]
    window = sd["preprocessor.featurizer.window"].numpy()
    save_file({"mel_fb": np.ascontiguousarray(fb.astype(np.float32)),
               "window": window.astype(np.float32)},
              model_dir / "mel_filters.safetensors")

    if aed:
        # tokenizer (aggregato o unificato) dai modelli SPE nel tar
        pieces, spl_size = canary_pieces(tar, names, ycfg)
        mynah["tokenizer"]["spl_size"] = spl_size
        # i token che il runtime risolverà per nome DEVONO esistere nel vocab
        pset = set(pieces)
        need = [mynah["decoder"]["eos_token"]]
        need += [f"<|{l}|>" for l in mynah["prompt"]["languages"]]
        need += [v for v in mynah["prompt"]["defaults"].values() if v]
        missing = [t for t in need if t not in pset]
        assert not missing, f"token assenti dal vocab: {missing}"
    else:
        # pieces dallo yaml (in ordine di id) + blank in coda
        vocab = ycfg.get("joint", {}).get("vocabulary") or ycfg["decoder"]["vocabulary"]
        pieces = list(vocab) + ["<blank>"]
    assert len(pieces) == mynah["decoder"]["vocab_size"]
    (model_dir / "tokens.json").write_text(json.dumps(pieces, ensure_ascii=False))
    (model_dir / "mynah.json").write_text(json.dumps(mynah, indent=1, ensure_ascii=False))
    print(f"OK {model_dir.name} [{mynah['engine']}] from .nemo: {len(tensors)} tensors "
          f"f32, {len(pieces)} pieces, mel fb {fb.shape} dal checkpoint")


def convert(model_dir: Path) -> None:
    if not (model_dir / "config.json").exists():
        nemos = sorted(model_dir.glob("*.nemo"))
        assert nemos, f"{model_dir}: ne' config.json (porting HF) ne' archivio .nemo"
        convert_from_nemo(model_dir, nemos[0])
        return
    cfg = json.loads((model_dir / "config.json").read_text())
    proc = json.loads((model_dir / "processor_config.json").read_text())
    tok_cfg = json.loads((model_dir / "tokenizer_config.json").read_text())
    fe = proc["feature_extractor"]

    builder = BUILDERS.get(cfg["model_type"])
    assert builder, f"model_type non supportato: {cfg['model_type']} (noti: {list(BUILDERS)})"
    assert fe["feature_size"] == cfg["encoder_config"]["num_mel_bins"]

    mynah = builder(model_dir, cfg, proc)
    vocab_size = mynah["decoder"]["vocab_size"]
    blank_id = mynah["decoder"]["blank_id"]

    pieces = load_pieces(model_dir, vocab_size, blank_id, tok_cfg.get("blank_token", "<blank>"))

    fb = melscale_fbanks(fe["n_fft"] // 2 + 1, 0.0, fe["sampling_rate"] / 2, fe["feature_size"], fe["sampling_rate"])
    window = np.hanning(fe["win_length"])  # simmetrica, come torch.hann_window(periodic=False)

    (model_dir / "mynah.json").write_text(json.dumps(mynah, indent=1, ensure_ascii=False))
    (model_dir / "tokens.json").write_text(json.dumps(pieces, ensure_ascii=False))
    save_file(
        {"mel_fb": fb.astype(np.float32), "window": window.astype(np.float32)},
        model_dir / "mel_filters.safetensors",
    )
    print(f"OK {model_dir.name} [{mynah['engine']}]: mynah.json, tokens.json ({len(pieces)} pieces), "
          f"mel_filters.safetensors (fb {fb.shape}, window {window.shape})")
    if "streaming" in mynah:
        print(f"   latency presets: {mynah['streaming']['att_context_presets']} "
              f"(default idx {mynah['streaming']['default_preset_index']})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    convert(Path(sys.argv[1]).resolve())
