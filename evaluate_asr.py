#!/usr/bin/env python3
"""
Benchmark a Wav2Vec2-CTC speech recognition model on the Pulaar maternal-health
test set, and compare the result with our reference evaluation.

Pipeline
--------
 1. Load the dataset (audio kept as raw bytes) + gender / dialect metadata
 2. Load your Wav2Vec2 model + processor (local folder or Hugging Face Hub id)
 3. For each sample: decode audio -> mono -> 16 kHz -> model -> greedy CTC decode
 4. Score: WER + CER overall, and broken down by gender and by dialect
 5. Save results.json / predictions.csv / breakdown.csv
 6. Print your scores next to our reference scores (reference/results.json)

Example
-------
    python evaluate_asr.py --model your-username/your-pulaar-model
"""

import argparse
import csv
import io
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import jiwer

# --------------------------------------------------------------------------- #
# Defaults - edit these two lines if the dataset id / column names change
# --------------------------------------------------------------------------- #
DEFAULT_DATASET = "guizme/Pulaar"      # TODO: replace with the maternal-health dataset id
DEFAULT_TEXT_COLUMN = "transcription"
MIN_GROUP_SIZE = 10                    # groups smaller than this get a warning


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="Local model folder OR Hugging Face Hub model id.")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="Hub dataset id (default: our maternal-health set).")
    p.add_argument("--config", default=None, help="Dataset config name, if any.")
    p.add_argument("--split", default="test", help="Split to evaluate (default: test).")
    p.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN, help="Column with the reference text.")
    p.add_argument("--gender-column", default="gender", help="Column with speaker gender.")
    p.add_argument("--dialect-column", default="dialect", help="Column with speaker dialect.")
    p.add_argument("--num-samples", type=int, default=None, help="Evaluate only N random samples (default: all).")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42).")
    p.add_argument("--normalize", action="store_true", help="Lowercase + strip punctuation before scoring.")
    p.add_argument("--reference", default="reference/results.json",
                   help="Our reference results to compare against (default: reference/results.json).")
    p.add_argument("--device", default=None, help="'cuda' or 'cpu' (default: auto).")
    p.add_argument("--output-dir", default="results", help="Output folder (default: ./results).")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Step 1 - Data
# --------------------------------------------------------------------------- #
def load_data(args):
    from datasets import Audio, ClassLabel, load_dataset

    ds = load_dataset(args.dataset, args.config, split=args.split).shuffle(seed=args.seed)
    if args.num_samples:
        ds = ds.select(range(min(args.num_samples, len(ds))))

    # Metadata columns used for the breakdowns. Missing columns are skipped.
    meta = {}
    for name, col in (("gender", args.gender_column), ("dialect", args.dialect_column)):
        if col not in ds.column_names:
            print(f"      (no '{col}' column in the dataset -> no {name} breakdown)")
            continue
        values = ds[col]
        feat = ds.features[col]
        if isinstance(feat, ClassLabel):               # ints -> readable labels
            values = [feat.int2str(v) for v in values]
        meta[name] = [str(v) if v not in (None, "") else "unknown" for v in values]

    # decode=False -> raw WAV bytes, decoded later with SoundFile (no TorchCodec needed)
    ds = ds.cast_column("audio", Audio(decode=False))
    return ds, meta


# --------------------------------------------------------------------------- #
# Step 2 & 3 - Model + inference
# --------------------------------------------------------------------------- #
def load_audio(audio_info, target_sr):
    """Raw bytes -> mono float waveform at the model's sampling rate."""
    import soundfile as sf
    import torch
    import torchaudio.functional as AF

    if audio_info.get("bytes"):
        wave, sr = sf.read(io.BytesIO(audio_info["bytes"]))
    else:
        wave, sr = sf.read(audio_info["path"])
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sr != target_sr:
        wave = AF.resample(torch.tensor(wave, dtype=torch.float32), sr, target_sr).numpy()
        sr = target_sr
    return wave, sr


def transcribe(ds, text_column, model_id, device_name):
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[2/6] Loading model '{model_id}' on {device}...")
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device).eval()
    target_sr = processor.feature_extractor.sampling_rate

    print("[3/6] Transcribing...")
    predictions, references = [], []
    with torch.no_grad():
        for i, example in enumerate(ds, start=1):
            wave, sr = load_audio(example["audio"], target_sr)
            inputs = processor(wave, sampling_rate=sr, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)          # greedy CTC decoding
            predictions.append(processor.batch_decode(pred_ids)[0])
            references.append(example[text_column])
            if i % 20 == 0 or i == len(ds):
                print(f"      {i}/{len(ds)}")
    return predictions, references


# --------------------------------------------------------------------------- #
# Step 4 - Metrics (overall + per group)
# --------------------------------------------------------------------------- #
def normalize_text(text):
    text = re.sub(r"[^\w\s]", "", text.lower())     # keeps letters such as ɓ ɗ ŋ ƴ
    return re.sub(r"\s+", " ", text).strip()


def score(refs, preds):
    """Corpus-level WER / CER (empty references are skipped: jiwer cannot score them)."""
    pairs = [(r, p) for r, p in zip(refs, preds) if r.strip()]
    if not pairs:
        return {"n": 0, "wer_percent": None, "cer_percent": None}
    r, p = zip(*pairs)
    return {"n": len(pairs),
            "wer_percent": round(jiwer.wer(list(r), list(p)) * 100, 2),
            "cer_percent": round(jiwer.cer(list(r), list(p)) * 100, 2)}


def breakdown(refs, preds, labels):
    """Score every distinct value of one metadata column (e.g. female / male)."""
    groups = defaultdict(lambda: ([], []))
    for ref, pred, label in zip(refs, preds, labels):
        groups[label][0].append(ref)
        groups[label][1].append(pred)
    return {label: score(r, p) for label, (r, p) in sorted(groups.items())}


# --------------------------------------------------------------------------- #
# Step 5 - Save
# --------------------------------------------------------------------------- #
def save_outputs(out_dir, args, refs, preds, meta, overall, groups, seconds):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "reference", "prediction", *meta.keys()])
        for i, (r, p) in enumerate(zip(refs, preds)):
            w.writerow([i + 1, r, p, *[meta[k][i] for k in meta]])

    with open(out / "breakdown.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group_by", "value", "n", "wer_percent", "cer_percent"])
        for by, vals in groups.items():
            for value, s in vals.items():
                w.writerow([by, value, s["n"], s["wer_percent"], s["cer_percent"]])

    summary = {"model": args.model, "dataset": args.dataset, "split": args.split,
               "normalized": args.normalize, "overall": overall, "breakdowns": groups,
               "seconds": round(seconds, 1)}
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


# --------------------------------------------------------------------------- #
# Step 6 - Report + comparison with our reference
# --------------------------------------------------------------------------- #
def fmt(x):
    return "  n/a" if x is None else f"{x:6.2f}"


def delta(mine, ref):
    if mine is None or ref is None:
        return ""
    d = mine - ref
    return f"{'+' if d > 0 else ''}{d:.2f}  ({'worse' if d > 0 else 'better' if d < 0 else 'same'})"


def report(summary, reference_path):
    ref = None
    if reference_path and Path(reference_path).exists():
        ref = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    else:
        print(f"\n(no reference file at '{reference_path}' -> showing your scores only)")

    print("\n=== RESULTS (WER %) ===")
    print(f"{'':<22}{'n':>5}{'yours':>9}{'ours':>9}   difference")
    rows = [("OVERALL", summary["overall"], ref["overall"] if ref else None)]
    for by, vals in summary["breakdowns"].items():
        for value, s in vals.items():
            r = ref["breakdowns"].get(by, {}).get(value) if ref else None
            rows.append((f"{by}={value}", s, r))
    for name, mine, theirs in rows:
        warn = "  [few samples]" if name != "OVERALL" and mine["n"] < MIN_GROUP_SIZE else ""
        their_wer = theirs["wer_percent"] if theirs else None
        print(f"{name:<22}{mine['n']:>5}{fmt(mine['wer_percent']):>9}{fmt(their_wer):>9}   "
              f"{delta(mine['wer_percent'], their_wer)}{warn}")
    print(f"\nCER overall: {summary['overall']['cer_percent']}%"
          + (f"   (ours: {ref['overall']['cer_percent']}%)" if ref else ""))


def main():
    args = parse_args()
    t0 = time.time()

    print(f"[1/6] Loading dataset '{args.dataset}' ({args.split})...")
    ds, meta = load_data(args)
    print(f"      {len(ds)} samples")

    preds, refs = transcribe(ds, args.text_column, args.model, args.device)

    print("[4/6] Computing metrics...")
    if args.normalize:
        preds = [normalize_text(p) for p in preds]
        refs = [normalize_text(r) for r in refs]
    overall = score(refs, preds)
    groups = {by: breakdown(refs, preds, labels) for by, labels in meta.items()}

    print(f"[5/6] Saving to '{args.output_dir}/'...")
    summary = save_outputs(args.output_dir, args, refs, preds, meta, overall, groups, time.time() - t0)

    print("[6/6] Report")
    report(summary, args.reference)


if __name__ == "__main__":
    main()
