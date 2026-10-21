#!/usr/bin/env python3
"""
Benchmark a Wav2Vec2-CTC speech recognition model on the Pulaar maternal-health
test set, and compare the result with our reference evaluation.

Pipeline
--------
 1. Load the dataset (audio kept as raw bytes) 
 2. Load your Wav2Vec2 model + processor 
 3. For each sample: decode audio -> mono -> 16 kHz -> model -> greedy CTC decode
 4. Score: WER + CER overall, and broken down by gender and by dialect

Example
-------
    python evaluate_asr.py --model your-username/your-pulaar-model
"""
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from jiwer import wer, cer
from pydub import AudioSegment
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import re
import tensorflow as tf
import unicodedata, unicodedata2

from transformers import AutoProcessor, AutoModelForCTC

 
def load_model(model_name, lang=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    if lang:  # MMS adapters
        processor.tokenizer.set_target_lang(lang)
        model.load_adapter(lang)
    model.to(device).eval()
    return processor, model, device
 
 
def ensure_wav_format(input_path):
    if input_path.lower().endswith(".wav"):
        return input_path
    audio = AudioSegment.from_file(input_path)
    wav_path = os.path.splitext(input_path)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path
 
 
def transcribe_array(audio_array, sampling_rate, processor, model, device):
    # stereo -> mono
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)
    audio_array = audio_array.astype("float32")
 
    # resample to 16 kHz
    if sampling_rate != 16000:
        audio_array = torchaudio.functional.resample(
            torch.from_numpy(audio_array), orig_freq=sampling_rate, new_freq=16000
        ).numpy()
 
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
 
    with torch.no_grad():
        logits = model(input_values).logits
 
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]
 
 
def normalize(text):
    """Lowercase, drop punctuation (keep accents/apostrophes), collapse spaces."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default="facebook/wav2vec2-base-960h")
    ap.add_argument("--audio_col", default="audio")
    ap.add_argument("--text_col", default="transcription")
    ap.add_argument("--audio_root", default="")
    ap.add_argument("--lang", default=None, help="MMS adapter language code, e.g. eng, fra")
    ap.add_argument("--out", default="predictions_wav2vec.csv")
    ap.add_argument("--sep", default=",")
    args = ap.parse_args()
 
    df = pd.read_csv(args.csv, sep=args.sep)
    processor, model, device = load_model(args.model, args.lang)
    print(f"Model: {args.model} | device: {device} | samples: {len(df)}")
 
    preds, refs, errors = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = os.path.join(args.audio_root, str(row[args.audio_col]))
        try:
            path = ensure_wav_format(path)
            audio, sr = sf.read(path)
            hyp = transcribe_array(audio, sr, processor, model, device)
        except Exception as e:
            hyp = ""
            errors.append((path, str(e)))
        preds.append(hyp)
        refs.append(str(row[args.text_col]))
 
    df["prediction"] = preds
    df["ref_norm"] = [normalize(r) for r in refs]
    df["pred_norm"] = [normalize(p) for p in preds]
 
    valid = df[df["ref_norm"].str.len() > 0].copy()
    valid["sample_wer"] = [wer(r, h) for r, h in zip(valid["ref_norm"], valid["pred_norm"])]
    valid["sample_cer"] = [cer(r, h) for r, h in zip(valid["ref_norm"], valid["pred_norm"])]
 
    corpus_wer = wer(valid["ref_norm"].tolist(), valid["pred_norm"].tolist())
    corpus_cer = cer(valid["ref_norm"].tolist(), valid["pred_norm"].tolist())
 
    valid.to_csv(args.out, index=False)
    print(f"\nEvaluated {len(valid)} samples ({len(errors)} failed to load)")
    print(f"WER: {corpus_wer:.4f} ({corpus_wer*100:.2f}%)")
    print(f"CER: {corpus_cer:.4f} ({corpus_cer*100:.2f}%)")
    print(f"Per-sample results saved to {args.out}")
    for p, e in errors[:5]:
        print(f"  ! {p}: {e}")
 
 
if __name__ == "__main__":
    main()
 
