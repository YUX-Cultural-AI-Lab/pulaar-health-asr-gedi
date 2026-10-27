"""
Benchmark several ASR models (wav2vec2 / MMS / Whisper) on the same Pular test set.

Model spec format:  NAME=TYPE:HF_ID[:LANG]
  TYPE = w2v      -> Wav2Vec2ForCTC (any CTC checkpoint)
  TYPE = mms      -> MMS adapter model, LANG is the MMS code (Fula = ful)
  TYPE = whisper  -> Whisper (LANG optional, see note below)

Example:
python benchmark_asr.py --csv D/test_.csv --n_samples 100 \
  --models \
  "Wav2Vec2-Fula=w2v:YUXCulturalAILab/wav2vec_pular_maternalhealth" \
  "ASR-Africa-Fula=w2v:your-org/asr-africa-fula-checkpoint" \
  "MMS-1b-all=mms:facebook/mms-1b-all:ful" \
  "Whisper-Large-v3=whisper:openai/whisper-large-v3" \
  "Whisper-Fula-FT=whisper:your-org/whisper-finetuned-pular"

NOTE: Whisper has no official Fula/Pular language token. For the base model,
leave LANG empty (auto-detect) or force a proxy (e.g. fr). Scores >100% WER
mean the model hallucinates extra words (insertions), which is expected here.
"""
import argparse
import gc
import os
import re
import unicodedata

import pandas as pd
import soundfile as sf
import torch
import torchaudio
from jiwer import cer, wer
from pydub import AudioSegment
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- audio / text
def ensure_wav_format(path):
    if path.lower().endswith(".wav"):
        return path
    wav_path = os.path.splitext(path)[0] + ".wav"
    if not os.path.exists(wav_path):
        AudioSegment.from_file(path).export(wav_path, format="wav")
    return wav_path


def load_audio_16k(path):
    """Load once -> mono float32 numpy array at 16 kHz."""
    path = ensure_wav_format(path)
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype("float32")
    if sr != 16000:
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio), orig_freq=sr, new_freq=16000
        ).numpy()
    return audio


def normalize(text):
    """Lowercase, drop punctuation (keep accents/apostrophes), collapse spaces."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- model loaders
class CTCModel:
    def __init__(self, model_id, lang=None):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
        if lang:  # MMS adapters
            self.processor.tokenizer.set_target_lang(lang)
            self.model.load_adapter(lang)
        self.model.to(DEVICE).eval()

    @torch.no_grad()
    def transcribe(self, audio):
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        logits = self.model(inputs.input_values.to(DEVICE)).logits
        ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(ids)[0]


class WhisperModel:
    def __init__(self, model_id, lang=None):
        from transformers import pipeline

        dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=dtype,
            device=DEVICE,
            chunk_length_s=30,  # handles clips > 30 s
        )
        self.gen_kwargs = {
            "task": "transcribe",
            "num_beams": 1,
            "no_repeat_ngram_size": 4,  # curbs repetition loops / hallucination
        }
        if lang:
            self.gen_kwargs["language"] = lang

    def transcribe(self, audio):
        # cap output length relative to duration to limit runaway hallucinations
        max_tokens = int(min(440, max(20, len(audio) / 16000 * 8)))
        out = self.pipe(
            {"raw": audio, "sampling_rate": 16000},
            generate_kwargs={**self.gen_kwargs, "max_new_tokens": max_tokens},
        )
        return out["text"]


def build_model(mtype, model_id, lang):
    if mtype == "w2v":
        return CTCModel(model_id)
    if mtype == "mms":
        return CTCModel(model_id, lang=lang or "ful")
    if mtype == "whisper":
        return WhisperModel(model_id, lang=lang or None)
    raise ValueError(f"Unknown model type: {mtype}")


def parse_spec(spec):
    """Accepts: NAME=TYPE:ID[:LANG] | TYPE:ID[:LANG] | ID (type guessed)."""
    name, rest = (spec.split("=", 1) + [None])[:2] if "=" in spec else (None, spec)
    if name is not None:
        name = name.strip()
    parts = rest.split(":")
    if parts[0] in ("w2v", "mms", "whisper") and len(parts) > 1:
        mtype, model_id = parts[0], parts[1]
        lang = parts[2] if len(parts) > 2 else None
    else:  # bare model id
        model_id, lang = parts[0], None
        mtype = "whisper" if "whisper" in model_id.lower() else "w2v"
    return name or model_id, mtype, model_id, lang


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="NAME=TYPE:HF_ID[:LANG]")
    ap.add_argument("--audio_col", default="audio")
    ap.add_argument("--text_col", default="transcription")
    ap.add_argument("--audio_root", default="")
    ap.add_argument("--n_samples", type=int, default=None, help="first N rows (use the same N for all models)")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv, sep=args.sep)
    if args.n_samples:
        df = df.head(args.n_samples).copy()
    df["ref_norm"] = df[args.text_col].map(normalize)

    # load audio once, reuse for every model
    audios, kept = [], []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Loading audio"):
        path = os.path.join(args.audio_root, str(row[args.audio_col]))
        try:
            audios.append(load_audio_16k(path))
            kept.append(True)
        except Exception as e:
            print(f"  ! {path}: {e}")
            audios.append(None)
            kept.append(False)
    df["_ok"] = kept

    summary = []
    for spec in args.models:
        name, mtype, model_id, lang = parse_spec(spec)
        print(f"\n=== {name} ({mtype}: {model_id}) | device: {DEVICE} ===")
        try:
            asr = build_model(mtype, model_id, lang)
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

        res = df.copy()
        res["prediction"] = preds
        res["pred_norm"] = res["prediction"].map(normalize)
        res = res[res["_ok"] & (res["ref_norm"].str.len() > 0)].copy()
        res["sample_wer"] = [wer(r, h) for r, h in zip(res["ref_norm"], res["pred_norm"])]
        res["sample_cer"] = [cer(r, h) for r, h in zip(res["ref_norm"], res["pred_norm"])]

        w = wer(res["ref_norm"].tolist(), res["pred_norm"].tolist()) * 100
        c = cer(res["ref_norm"].tolist(), res["pred_norm"].tolist()) * 100
        summary.append((name, len(res), w, c))
        res.drop(columns=["_ok"]).to_csv(
            os.path.join(args.out_dir, f"predictions_{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}.csv"),
            index=False,
        )

        # free GPU memory before the next model
        del asr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------ summary table
    line = "-" * 59
    print("\n" + line)
    print(f"{'Model  architecture':<36}{'samples':>7}{'wer':>9}{'cer':>8}")
    print(line)
    for name, n, w, c in summary:
        print(f"{name:<36}{n:>7}{w:>9.2f}{c:>8.2f}")
    print(line)
    pd.DataFrame(summary, columns=["model", "samples", "WER", "CER"]).to_csv(
        os.path.join(args.out_dir, "summary.csv"), index=False
    )
    print(f"Saved summary + per-model predictions in {args.out_dir}/")


if __name__ == "__main__":
    main()
