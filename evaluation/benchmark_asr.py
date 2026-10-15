
import argparse
import re
from pathlib import Path

import librosa
import pandas as pd
import torch
from jiwer import wer, cer
from transformers import pipeline


# Previously reported results.
# Verify that these use the same test set and normalization
# before treating them as directly comparable.
BASELINES = [
    {
        "model": "Wav2Vec2-Fula",
        "architecture": "Wav2Vec2",
        "wer": 43.81,
        "cer": 14.14,
        "type": "Reported baseline",
    },
    {
        "model": "ASR-Africa-Fula",
        "architecture": "Wav2Vec2",
        "wer": 91.24,
        "cer": 32.88,
        "type": "Reported baseline",
    },
    {
        "model": "Whisper-Large-v3",
        "architecture": "Whisper",
        "wer": 124.02,
        "cer": 61.38,
        "type": "Reported baseline",
    },
]


def normalize_text(text):
    """Basic, consistent normalization for references and predictions."""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def main():
    parser = argparse.ArgumentParser(
        description="Reusable maternal-health ASR benchmark"
    )
    parser.add_argument("--model", required=True,
                        help="Hugging Face model ID or local model path")
    parser.add_argument("--dataset-csv", required=True,
                        help="CSV manifest containing audio paths and references")
    parser.add_argument("--audio-column", default="audio_path")
    parser.add_argument("--text-column", default="reference")
    parser.add_argument("--language-column", default="language")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Hide the previously reported baseline table")
    args = parser.parse_args()

    manifest = pd.read_csv(args.dataset_csv)

    for column in (args.audio_column, args.text_column):
        if column not in manifest.columns:
            raise ValueError(f"Missing required CSV column: {column}")

    # Keep only rows with valid paths and reference transcriptions.
    manifest = manifest.dropna(
        subset=[args.audio_column, args.text_column]
    ).copy()

    manifest[args.audio_column] = manifest[args.audio_column].astype(str)
    manifest[args.text_column] = manifest[args.text_column].astype(str)

    missing = [
        path for path in manifest[args.audio_column]
        if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} audio files were not found. "
            f"Example: {missing[0]}"
        )

    # Use a fixed sample order so repeated runs use the same examples.
    manifest = manifest.head(args.num_samples).reset_index(drop=True)

    if manifest.empty:
        raise ValueError("No valid evaluation samples found.")

    device = 0 if torch.cuda.is_available() else -1

    print(f"Model: {args.model}")
    print(f"Samples: {len(manifest)}")
    print(f"Device: {'CUDA GPU' if device == 0 else 'CPU'}")

    recognizer = pipeline(
        task="automatic-speech-recognition",
        model=args.model,
        device=device,
    )

    records = []

    for index, row in manifest.iterrows():
        audio_path = row[args.audio_column]
        reference = row[args.text_column]

        try:
            # Resample audio consistently to 16 kHz mono.
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)

            prediction = recognizer({
                "raw": audio,
                "sampling_rate": 16000,
            })["text"]

            records.append({
                "audio_path": audio_path,
                "reference": reference,
                "prediction": prediction,
                "reference_normalized": normalize_text(reference),
                "prediction_normalized": normalize_text(prediction),
                "language": (
                    row[args.language_column]
                    if args.language_column in manifest.columns
                    else "unspecified"
                ),
            })

            print(f"[{index + 1}/{len(manifest)}] "
                  f"{prediction[:100]}")

        except Exception as exc:
            print(f"Skipping {audio_path}: {exc}")

    if not records:
        raise RuntimeError("No audio samples were evaluated successfully.")

    predictions = pd.DataFrame(records)
    references = predictions["reference_normalized"].tolist()
    hypotheses = predictions["prediction_normalized"].tolist()

    # jiwer reports error rates as fractions; convert to percentages.
    overall_wer = 100 * wer(references, hypotheses)
    overall_cer = 100 * cer(references, hypotheses)

    # Report aggregate and per-language scores.
    print("\n=== OVERALL RESULTS ===")
    print(f"Model: {args.model}")
    print(f"Evaluated samples: {len(predictions)}")
    print(f"WER: {overall_wer:.2f}%")
    print(f"CER: {overall_cer:.2f}%")

    language_results = []
    for language, group in predictions.groupby("language"):
        lang_wer = 100 * wer(
            group["reference_normalized"].tolist(),
            group["prediction_normalized"].tolist(),
        )
        lang_cer = 100 * cer(
            group["reference_normalized"].tolist(),
            group["prediction_normalized"].tolist(),
        )

        language_results.append({
            "model": args.model,
            "language": language,
            "samples": len(group),
            "wer": round(lang_wer, 2),
            "cer": round(lang_cer, 2),
        })

    print("\n=== RESULTS BY LANGUAGE ===")
    print(pd.DataFrame(language_results).to_string(index=False))

    output_dir = Path(args.output_dir)
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions for error analysis.
    safe_model_name = re.sub(r"[^\w.-]+", "_", args.model)
    predictions.to_csv(
        prediction_dir / f"{safe_model_name}_predictions.csv",
        index=False,
    )

    # Append aggregate results so multiple models can be compared.
    result_file = output_dir / "benchmark_results.csv"
    summary = pd.DataFrame([{
        "model": args.model,
        "architecture": "Hugging Face ASR pipeline",
        "samples": len(predictions),
        "wer": round(overall_wer, 2),
        "cer": round(overall_cer, 2),
        "type": "New evaluation",
    }])

    summary.to_csv(
        result_file,
        mode="a",
        header=not result_file.exists(),
        index=False,
    )

    # Display the reported baselines alongside completed runs.
    current_results = pd.read_csv(result_file)
    comparison = pd.concat(
        [
            pd.DataFrame([]) if args.no_baselines
            else pd.DataFrame(BASELINES),
            current_results,
        ],
        ignore_index=True,
    )

    print("\n=== MODEL COMPARISON ===")
    print(comparison.to_string(index=False))
    print(f"\nResults saved to: {result_file}")
    print(f"Predictions saved to: {prediction_dir}")


if __name__ == "__main__":
    main()
