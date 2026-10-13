# Beyond Accuracy: Building Inclusive Speech Recognition for Pulaar Through a Gender and Dialect Lens

## Overview

ASR can improve maternal healthcare access, but limited data and linguistic diversity make it difficult for systems to accurately recognize African languages and dialects. Overall accuracy may also hide disparities across genders and regional varieties, such as Fuuta Tooro and Fuuta Jallon Pulaar. Technical accuracy alone is not enough: inclusive AI requires active community participation to ensure systems reflect users’ lived experiences, priorities, and needs.

This project explores the development and evaluation of ASR for Pulaar maternal and reproductive health communication, with a focus on **Gender, Equity, Diversity, and Inclusion (GEDI)**.

## Objectives

- Adapt open-source ASR models to Pulaar maternal and reproductive health speech.
- Evaluate speech recognition across the Fuuta Tooro and Fuuta Jallon dialects.
- Examine potential disparities related to gender, dialect, accent, and code-switching.
- Assess usability and accessibility in realistic use conditions.

## Methodology

The project followed a GEDI-informed approach throughout the ASR development and evaluation process.

### 1. Data Collection and Validation

- We collected maternal and reproductive health utterances from **40 Pulaar speakers**, covering the Fouta Toro and Fouta Djallion dialects and including women with disabilities.

- Data validation involved linguistic expertise, translation review, and maternal-health specialists to improve the accuracy and contextual relevance of the collected utterances.



### 2. Model Benchmarking and Fine-Tuning

Open-source ASR models were benchmarked on Pulaar health-related speech using Word Error Rate (WER) and Character Error Rate (CER). A selected model was fine-tuned on the collected data to improve recognition of domain-specific vocabulary and dialectal variation.


### 3. GEDI-Informed Evaluation

The evaluation considered potential recognition challenges across:

- **Gender:** Differences in transcription performance between speaker groups.
- **Dialect:** Recognition of regional expressions and linguistic variation.
- **Accent:** Differences in pronunciation and speaking patterns.
- **Code-switching:** Speech that mixes Pulaar with French or other languages.

### 4. Participatory User Testing

User testing involved approximately 10 participants across diverse speaker groups and dialects in Senegal, including community health workers and linguists.

Testing considered transcription quality, dialectal fidelity, and platform usability in quiet and noisy environments. Participants rated the evaluated aspects on a scale from 1 (Very Poor) to 5 (Excellent).

We provide:

1. **A test dataset** of 200 maternal-health audio samples with transcriptions,
   speaker **gender** and **dialect**: <DATASET_LINK>
2. **Our reference results**, obtained with our own model
   ([`YUXCulturalAILab/wav2vec_pular_maternalhealth`](https://huggingface.co/YUXCulturalAILab/wav2vec_pular_maternalhealth)),
   in [`reference/`](reference/).
3. **An evaluation script** so you can run *your* model on the same data and
   compare your scores with ours, overall and by gender and dialect.
4. **A user testing and validation protocol (Stage 6)** with our results from real
   users, including Community Health Workers and linguists: [`user_testing/`](user_testing/).

## Two layers of evaluation

| Layer | What it tells you | Where |
|-------|-------------------|-------|
| **Automatic benchmark** | WER / CER against reference transcriptions, overall and by gender and dialect | this page + `evaluate_asr.py` |
| **Stage 6: User testing & validation** | Whether the transcriptions are accurate, meaningful, usable and faithful to the dialect for real users (GEDI lens) | [`user_testing/`](user_testing/README.md) |

### Stage 6 results at a glance

| Dimension | Score |
|-----------|-------|
| ASR quality | 4 / 5 |
| Dialect and linguistic fidelity | 3 / 5 |
| Platform usability | 5 / 5 |
| **Overall** | **4 / 5** |

Usability and ASR quality were rated well. The main gap is **dialect and linguistic
diversity**, which is why the automatic benchmark reports results by dialect.
Details, participants and protocol: [`user_testing/README.md`](user_testing/README.md).



## Key Results

| Evaluation measure | Result |
|---|---:|
| Baseline WER | Approximately 43% |
| Fine-tuned WER | Approximately 28% |
| Absolute WER improvement | 15 percentage points |
| User-rated transcription quality | 4/5 |
| User-rated dialect and linguistic fidelity | 3/5 |
| User-rated platform usability | 4/5 |



These results indicate improved transcription performance after fine-tuning. User testing also highlighted persistent challenges with dialect-specific expressions, accents, and spelling conventions.

**Note:** The reported WER values should be interpreted in relation to the evaluation dataset and experimental setup. Overall performance does not necessarily imply equal accuracy across all speaker groups.

## GEDI Contribution

This project treats inclusion as part of the ASR development and evaluation process rather than as a final checklist.

The approach considers:

- Representation of diverse speakers and dialects.
- Linguistic and contextual validation of health-related utterances.
- Potential disparities hidden by aggregate performance metrics.
- Accessibility and usability for underserved communities.
- Responsible handling of audio recordings and participant information.

The aim is to complement technical benchmarking with evaluation that considers whose speech is recognized, where recognition challenges remain, and how the system performs in the contexts in which it may be used.

## Repository Structure

```text
pulaar-health-asr-gedi/
├── docs/
│   ├── methodology.md
│   ├── gedi_framework.md
│   └── data_collection.md
├── src/
│   ├── preprocessing/
│   └── training/
├── evaluation/
│   ├── benchmarking.py
│   ├── metrics.py
│   └── gedi_evaluation.md
├── results/
│   └── benchmark_results.csv
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

*This structure describes the proposed organization of the repository. Files should be added as the corresponding code and documentation become available.*


# Pulaar ASR Evaluation (Fouta Toro dialect)

A single script that evaluates a fine-tuned **Wav2Vec2-CTC** speech recognition
model on Pulaar test data and reports **WER** (Word Error Rate) and **CER**
(Character Error Rate).

## Quick start

```bash
git clone <your-repo-url>
cd pulaar-asr-eval
pip install -r requirements.txt

# Evaluate on the Pulaar test set
python evaluate_asr.py --model <YUXCulturalAILab/wav2vec_pular_maternalhealth> --dataset pulaar

# Evaluate on Google FLEURS (Fula, Senegal), 50 random samples
python evaluate_asr.py --model <YUXCulturalAILab/wav2vec_pular_maternalhealth> --dataset fleurs --num-samples 50
```

If the model or dataset is private, log in once with `huggingface-cli login`
(or set the `HF_TOKEN` environment variable).

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | YUXCulturalAILab/wav2vec_pular_maternalhealth | – |
| `--dataset` | `pulaar`, `fleurs`, or any Hub dataset id | `pulaar` |
| `--config` | Dataset config (e.g. `ff_sn`) | preset |
| `--split` | Split to evaluate | `test` |
| `--text-column` | Column with the reference text | `transcription` |
| `--num-samples` | 50 samples | or whole split |
| `--seed` | Shuffle seed (reproducibility) | `42` |
| `--normalize` | Lowercase + remove punctuation before scoring | off |
| `--device` | `cuda` or `cpu` | auto |
| `--output-dir` | Output folder | `results/` |

## The pipeline

```
 Hugging Face dataset            Wav2Vec2-CTC model
 (audio + transcription)         (local folder or Hub)
          │                               │
          ▼                               │
  1. Load split, shuffle (seed),          │
     keep audio as raw WAV bytes          │
          │                               │
          ▼                               ▼
  2. Decode bytes with SoundFile ──► 3. Preprocess
     → mono → resample to 16 kHz        (Wav2Vec2Processor: normalise waveform)
                                              │
                                              ▼
                                      4. Forward pass → logits
                                              │
                                              ▼
                                      5. Greedy CTC decoding
                                         (argmax per frame, collapse repeats,
                                          remove blanks) → text
                                              │
          reference transcription ────────────┤
                                              ▼
                                      6. WER / CER (jiwer)
                                              │
                                              ▼
                                   results/results.json
                                   results/predictions.csv
```

**Step by step**

1. **Data loading.** Only the requested split is downloaded. Audio is loaded with
   `decode=False` so we get the raw WAV bytes and decode them ourselves with
   SoundFile (this avoids the TorchCodec requirement of recent `datasets`).
2. **Audio preparation.** Stereo is averaged to mono and the signal is resampled to
   the rate the model was trained on (16 kHz for Wav2Vec2).
3. **Feature extraction.** `Wav2Vec2Processor` normalises the waveform into model inputs.
4. **Inference.** The model outputs, for every ~20 ms audio frame, a probability
   distribution over the character vocabulary (logits).
5. **Decoding.** Greedy CTC decoding: take the most likely token per frame, merge
   consecutive duplicates, drop the blank token. No language model is used.
6. **Scoring.**
   - **WER** = (substitutions + deletions + insertions) / number of reference words.
     It can exceed 100% when the model outputs many wrong extra words.
   - **CER** = same idea at character level. It is usually more informative for a
     language with a young standard orthography like Pulaar, where a one-letter
     spelling difference (`pulaagu` vs `puraagu`) counts as a whole wrong word in WER.

## Outputs

- `results/results.json` – model, dataset, number of samples, WER, CER, runtime.
- `results/predictions.csv` – reference, prediction and per-sample WER for every
  example, handy for error analysis.

## Results

| Dataset | Samples | WER |
|---------|---------|-----|
| `guizme/Pulaar` (test) | 100 | --xx-- |
| `google/fleurs` `ff_sn` (test) | 100 | _xx_ |


# Pulaar ASR Benchmark: Maternal Health

A small benchmark for Pulaar (Fouta Toro) speech recognition in the maternal-health domain.

We provide:

1. **A test dataset** of 200 maternal-health audio samples with transcriptions,
   speaker **gender** and **dialect**: <DATASET_LINK>
2. **Our reference results**, obtained with our own model
   ([`YUXCulturalAILab/wav2vec_pular_maternalhealth`](https://huggingface.co/YUXCulturalAILab/wav2vec_pular_maternalhealth)),
   in [`reference/`](reference/).
3. **An evaluation script** so you can run *your* model on the same data and
   compare your scores with ours, overall and by gender and dialect.

## Reference results (ours)

| Group | Samples | WER % | CER % |
|-------|---------|-------|-------|
| **Overall** | 200 | _fill in_ | _fill in_ |
| Gender: female | _ | _ | _ |
| Gender: male | _ | _ | _ |
| Dialect: _name_ | _ | _ | _ |
| Dialect: _name_ | _ | _ | _ |

Full numbers: [`reference/results.json`](reference/results.json) and
[`reference/predictions.csv`](reference/predictions.csv).

## Evaluate your own model

Your model must be a **Wav2Vec2 model with a CTC head** (`Wav2Vec2ForCTC`),
stored in a local folder or on the Hugging Face Hub.

```bash
git clone <your-repo-url>
cd pulaar-asr-eval
pip install -r requirements.txt

python evaluate_asr.py --model your-username/your-pulaar-model
# or a local folder:
python evaluate_asr.py --model ./path/to/your/model
```

The dataset is downloaded automatically. If something is private, run
`huggingface-cli login` first (or set `HF_TOKEN`).

### What you get

The script prints your scores next to ours:

```
=== RESULTS (WER %) ===
                          n    yours     ours   difference
OVERALL                 200    xx.xx    xx.xx   +x.xx  (worse)
gender=female           xx    xx.xx    xx.xx   ...
gender=male             xx    xx.xx    xx.xx   ...
dialect=...             xx    xx.xx    xx.xx   ...
```

and saves, in `results/`:

| File | Content |
|------|---------|
| `results.json` | Overall + per-gender + per-dialect WER/CER |
| `breakdown.csv` | Same breakdown as a table |
| `predictions.csv` | Reference vs prediction for every sample, with gender and dialect |

Groups with fewer than 10 samples are flagged `[few samples]`. Their scores
are noisy, so read them with care.

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Local folder or Hub model id (**required**) | – |
| `--dataset` | Hub dataset id | our maternal-health set |
| `--split` | Split to evaluate | `test` |
| `--text-column` | Column with reference text | `transcription` |
| `--gender-column` | Column with speaker gender | `gender` |
| `--dialect-column` | Column with speaker dialect | `dialect` |
| `--num-samples` | Random subset (quick test) | all |
| `--normalize` | Lowercase and strip punctuation before scoring | off |
| `--reference` | Reference results to compare with | `reference/results.json` |
| `--device` | `cuda` or `cpu` | auto |
| `--output-dir` | Output folder | `results/` |
| `--seed` | Shuffle seed | `42` |

> For a fair comparison with our numbers, run with the default settings
> (no `--normalize`, no `--num-samples`).

## The pipeline

```
 Maternal-health dataset                    Your Wav2Vec2-CTC model
 (audio, transcription, gender, dialect)    (local folder or Hub)
          │                                          │
          ▼                                          │
  1. Load split, keep audio as raw bytes,            │
     read gender + dialect labels                    │
          │                                          │
          ▼                                          ▼
  2. Decode with SoundFile ─────────────────► 3. Preprocess
     → mono → resample to 16 kHz                (Wav2Vec2Processor)
                                                     │
                                                     ▼
                                             4. Forward pass → logits
                                                     │
                                                     ▼
                                             5. Greedy CTC decoding → text
                                                     │
        reference transcription ─────────────────────┤
                                                     ▼
                                      6. WER / CER with jiwer
                                         ├─ overall
                                         ├─ by gender
                                         └─ by dialect
                                                     │
                                                     ▼
                              7. Compare with reference/results.json
                                 + save results/*.json, *.csv
```

1. **Data loading.** Audio is read as raw WAV bytes (`decode=False`) and decoded
   with SoundFile, which avoids the TorchCodec requirement of recent `datasets`
   versions. Gender and dialect labels are read from the dataset columns.
2. **Audio preparation.** Stereo is averaged to mono, then resampled to the
   model's sampling rate (16 kHz).
3. **Feature extraction.** `Wav2Vec2Processor` normalises the waveform.
4. **Inference.** The model gives, for every ~20 ms frame, scores over the
   character vocabulary.
5. **Decoding.** Greedy CTC: take the best token per frame, merge repeats, drop
   blanks. No language model is used, for fairness and simplicity.
6. **Scoring.**
   - **WER** = (substitutions + deletions + insertions) / reference words.
     It can exceed 100%.
   - **CER** = the same at character level. It is gentler on spelling variation
     (e.g. `pulaagu` vs `puraagu` is one wrong word in WER but one wrong character in CER).
   - Scores are computed over all words of a group (not averaged per sentence),
     separately for each gender value and each dialect value.
7. **Comparison.** Your scores are printed next to the reference scores, with the
   difference. Negative means better than ours.

## Reproducing our reference results

```bash
python evaluate_asr.py --model YUXCulturalAILab/wav2vec_pular_maternalhealth \
    --output-dir reference
```

## Notes

- Do not train on the test set. Doing so makes the comparison meaningless.
- Results depend on spelling conventions, so compare WER and CER together.
- Don't commit model weights or tokens. Host your model on the Hugging Face Hub.


## Responsible AI and Data Ethics

Maternal and reproductive health recordings can contain sensitive personal information. Raw audio, identifiable transcripts, and participant details should not be published without appropriate consent and authorization.

Any shared datasets, model weights, or evaluation outputs must comply with the relevant research ethics requirements, participant consent, privacy protections, and project data-sharing agreements.

ASR-generated text may contain errors and should be reviewed by a qualified person before being used to inform healthcare decisions.

## Current Limitations

- The evaluation covers two Pulaar dialects and does not represent the full diversity of Pulaar-speaking communities.
- User testing involved a relatively small number of participants.
- Dialect and linguistic fidelity remain areas for improvement.
- Further disaggregated evaluation is needed to establish whether performance differs systematically across gender, dialect, accent, and other relevant factors.

## Future Work

- Expand linguistic coverage and representation of underserved speaker groups.
- Improve recognition of dialect-specific health terminology and code-switching.
- Strengthen disaggregated evaluation across relevant speaker groups.
- Extend participatory testing and accessibility assessment.
- Release approved code, documentation, and research outputs to support reproducibility and further research.

## Acknowledgements

This work was conducted under the GRAIN EmpowerHerAI initiative. The project involved collaboration with linguistic and maternal-health experts and contributors to Pulaar data collection and validation.

Partner acknowledgements and institutional affiliations should be finalized in accordance with project agreements.

## Citation

If you use this repository in your research, please cite the associated publication once available.

A `CITATION.cff` file can be added when the authors, publication details, and preferred citation format are confirmed.

## License

The repository license will be specified after confirming the ownership and sharing permissions for the code and documentation. Dataset and model licenses may differ from the repository's software license.

