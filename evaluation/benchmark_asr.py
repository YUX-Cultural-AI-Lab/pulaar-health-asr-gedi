"""
Benchmark ASR models (wav2vec2 / MMS / Whisper) on a Hugging Face dataset.

Usage:
  python evaluation/benchmark_asr.py \
      --model "YUXCulturalAILab/wav2vec_pular_maternalhealth" \
      --dataset "YUXCulturalAILab/maternal-health-Asr-pular" \
      --num-samples 200

Several models at once (same samples for each):
  python evaluation/benchmark_asr.py --dataset <ds> --num-samples 100 \
      --model "YUXCulturalAILab/wav2vec_pular_maternalhealth" \
              "AIHubSN/M-Kiriku-ASR" \
              "MMS=mms:facebook/mms-1b-all:ful"

Model spec:  ID   |   NAME=TYPE:ID[:LANG]     (TYPE = w2v | mms | whisper)
If TYPE is not given it is auto-detected from the model's config.json.

Results: each new model is added as a row to results/benchmarks_results.csv
(columns: model, samples, wer, cer). Re-running a model with the same name
replaces its row. Per-sample predictions go to results/predictions/.

Private/gated repos: set HF_TOKEN in the environment or pass --hf-token.
"""
import argparse
import gc
import io
import os
import re
import unicodedata

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from datasets import Audio, load_dataset
from jiwer import cer, wer
from tqdm.auto import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
TEXT_CANDIDATES = ["transcription", "text", "sentence", "transcript", "normalized_text"]

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ------------------------------------------------------------------ text/audio
def normalize(text):
    """Lowercase, drop punctuation (keep accents/apostrophes), collapse spaces."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def decode_audio_16k(audio_dict):
    """audio_dict = {'bytes': ..., 'path': ...} (Audio(decode=False)) -> mono float32 @16k."""
    raw, path = audio_dict.get("bytes"), audio_dict.get("path")
    try:
        audio, sr = sf.read(io.BytesIO(raw) if raw is not None else path)
    except Exception:  # formats libsndfile can't read (e.g. some mp3/m4a)
        from pydub import AudioSegment

        seg = AudioSegment.from_file(io.BytesIO(raw) if raw is not None else path)
        audio = np.array(seg.get_array_of_samples(), dtype=np.float32)
        audio = audio.reshape(-1, seg.channels) / float(1 << (8 * seg.sample_width - 1))
        sr = seg.frame_rate
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype("float32")
    if sr != 16000:
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio), orig_freq=sr, new_freq=16000
        ).numpy()
    return audio


# ---------------------------------------------------------------------- models
class CTCModel:
    def __init__(self, model_id, lang=None, token=None):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.processor = Wav2Vec2Processor.from_pretrained(model_id, token=token)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id, token=token)
        if lang:  # MMS adapters
            self.processor.tokenizer.set_target_lang(lang)
            self.model.load_adapter(lang)
        self.model.to(DEVICE).eval()

    @torch.no_grad()
    def transcribe(self, audio):
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        logits = self.model(inputs.input_values.to(DEVICE)).logits
        return self.processor.batch_decode(torch.argmax(logits, dim=-1))[0]


class WhisperModel:
    def __init__(self, model_id, lang=None, token=None):
        from transformers import pipeline

        dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
        self.pipe = pipeline(
            "automatic-speech-recognition", model=model_id, torch_dtype=dtype,
            device=DEVICE, chunk_length_s=30, token=token,
        )
        self.gen_kwargs = {"task": "transcribe", "num_beams": 1, "no_repeat_ngram_size": 4}
        if lang:
            self.gen_kwargs["language"] = lang

    def transcribe(self, audio):
        max_tokens = int(min(440, max(20, len(audio) / 16000 * 8)))
        out = self.pipe(
            {"raw": audio, "sampling_rate": 16000},
            generate_kwargs={**self.gen_kwargs, "max_new_tokens": max_tokens},
        )
        return out["text"]


def detect_type(model_id, token=None):
    """Whisper vs CTC, from config.json (falls back to the name)."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, token=token)
        return "whisper" if cfg.model_type == "whisper" else "w2v"
    except Exception:
        return "whisper" if "whisper" in model_id.lower() else "w2v"


def parse_spec(spec, token=None):
    """ID | NAME=TYPE:ID[:LANG] | TYPE:ID[:LANG]"""
    name, rest = (spec.split("=", 1) if "=" in spec else (None, spec))
    parts = rest.split(":")
    if parts[0] in ("w2v", "mms", "whisper") and len(parts) > 1:
        mtype, model_id = parts[0], parts[1]
        lang = parts[2] if len(parts) > 2 else None
    else:
        model_id, lang = parts[0], None
        mtype = detect_type(model_id, token)
    return (name or model_id).strip(), mtype, model_id, lang


def build_model(mtype, model_id, lang, token):
    if mtype == "w2v":
        return CTCModel(model_id, token=token)
    if mtype == "mms":
        return CTCModel(model_id, lang=lang or "ful", token=token)
    if mtype == "whisper":
        return WhisperModel(model_id, lang=lang, token=token)
    raise ValueError(f"Unknown model type: {mtype}")


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", "--models", nargs="+", required=True, dest="models")
    ap.add_argument("--dataset", required=True, help="HF dataset id, e.g. org/name")
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--split", default="test")
    ap.add_argument("--config", default=None, help="dataset config/subset name, if any")
    ap.add_argument("--audio-col", default=None, help="auto-detected if omitted")
    ap.add_argument("--text-col", default=None, help="auto-detected if omitted")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--results-csv", default=None,
                    help="default: <out-dir>/benchmarks_results.csv")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    token = args.hf_token
    pred_dir = os.path.join(args.out_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    results_csv = args.results_csv or os.path.join(args.out_dir, "benchmarks_results.csv")

    # ---- dataset
    print(f"Loading dataset {args.dataset} [{args.split}] ...")
    ds = load_dataset(args.dataset, args.config, split=args.split, token=token)

    audio_col = args.audio_col or next(
        (k for k, v in ds.features.items() if isinstance(v, Audio)), None
    )
    text_col = args.text_col or next((c for c in TEXT_CANDIDATES if c in ds.column_names), None)
    if audio_col is None or text_col is None:
        raise SystemExit(
            f"Could not detect columns (found {ds.column_names}). "
            "Pass --audio-col and --text-col."
        )
    print(f"audio column: {audio_col} | text column: {text_col} | rows in split: {len(ds)}")

    ds = ds.select(range(min(args.num_samples, len(ds))))
    ds = ds.cast_column(audio_col, Audio(decode=False))  # decode ourselves (no torchcodec needed)

    refs, audios = [], []
    for row in tqdm(ds, desc="Loading audio"):
        try:
            audios.append(decode_audio_16k(row[audio_col]))
        except Exception as e:
            print(f"  ! audio error: {e}")
            audios.append(None)
        refs.append(row[text_col])
    base = pd.DataFrame({"reference": refs})
    base["ref_norm"] = base["reference"].map(normalize)
    ok = np.array([a is not None for a in audios])

    # ---- models
    summary = []
    for spec in args.models:
        name, mtype, model_id, lang = parse_spec(spec, token)
        print(f"\n=== {name} ({mtype}: {model_id}) | device: {DEVICE} ===")
        try:
            asr = build_model(mtype, model_id, lang, token)
        except Exception as e:
            print(f"  ! could not load {name}: {e}")
            summary.append((name, 0, float("nan"), float("nan")))
            continue

        preds = []
        for audio in tqdm(audios, desc=name):
            if audio is None:
                preds.append("")
                continue
            try:
                preds.append(asr.transcribe(audio))
            except Exception as e:
                print(f"  ! inference error: {e}")
                preds.append("")

        res = base.copy()
        res["prediction"] = preds
        res["pred_norm"] = res["prediction"].map(normalize)
        res = res[ok & (res["ref_norm"].str.len() > 0).to_numpy()].copy()
        res["sample_wer"] = [wer(r, h) for r, h in zip(res["ref_norm"], res["pred_norm"])]
        res["sample_cer"] = [cer(r, h) for r, h in zip(res["ref_norm"], res["pred_norm"])]

        w = wer(res["ref_norm"].tolist(), res["pred_norm"].tolist()) * 100
        c = cer(res["ref_norm"].tolist(), res["pred_norm"].tolist()) * 100
        summary.append((name, len(res), w, c))
        res.to_csv(os.path.join(pred_dir, f"predictions_{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}.csv"), index=False)

        del asr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- merge into the benchmark table (append new rows, replace same-name rows)
    cols = ["model", "samples", "wer", "cer"]
    new = pd.DataFrame(summary, columns=cols)
    if os.path.exists(results_csv):
        old = pd.read_csv(results_csv)
        old = old[~old["model"].isin(new["model"])]
        table = pd.concat([old, new], ignore_index=True)
    else:
        table = new
    table.to_csv(results_csv, index=False)

    new_names = set(new["model"])
    line = "-" * 59
    print("\n" + line)
    print(f"{'Model  architecture':<36}{'samples':>7}{'wer':>9}{'cer':>8}")
    print(line)
    for _, r in table.iterrows():
        tag = "  <- new" if r["model"] in new_names else ""
        print(f"{r['model']:<36}{int(r['samples']):>7}{r['wer']:>9.2f}{r['cer']:>8.2f}{tag}")
    print(line)
    print(f"Table saved to {results_csv}")


if __name__ == "__main__":
    main()
