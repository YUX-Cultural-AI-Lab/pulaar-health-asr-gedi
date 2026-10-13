# Beyond Accuracy: Building Inclusive Speech Recognition for Pulaar Through a Gender and Dialect Lens

## Overview

ASR can improve maternal healthcare access, but limited data and linguistic diversity make it difficult for systems to accurately recognize African languages and dialects. Overall accuracy may also hide disparities across genders and regional varieties, such as Fouta Toro and Fouta Jallon Pulaar. Technical accuracy alone is not enough: inclusive AI requires active community participation to ensure systems reflect users’ lived experiences, priorities, and needs.

This project explores the development and evaluation of ASR for Pulaar maternal and reproductive health communication, with a focus on **Gender, Equity, Diversity, and Inclusion (GEDI)**.

## Objectives

- Adapt open-source ASR models to Pulaar maternal and reproductive health speech.
- Evaluate speech recognition across the Fouta Toro and Fouta Djalion dialects.
- Examine potential disparities related to gender, dialect, accent, and code-switching.
- Assess usability and accessibility in realistic use conditions.
- Test and apply GEDI Matrix throughout the the project  

## Methodology

The project followed a GEDI-informed approach throughout the ASR development and evaluation process.The framework assesses where innovation might unintentionally exclude or harm the populations it aims to serve. It helped us identify gaps across women, communities, dialects, and regions, using these insights to improve the system and embed the GRAIN GEDI “Design-by-Inclusion” Standard throughout the ASR engine lifecycle.

### Stage 1 & 2 : Community Engagement & Data collection

- The initial phase focused on deeply understanding the target community, assessing local access to technology, and aligning our research objectives with real-world needs. A central question guided this stage: Are we addressing a genuine community priority, a technical research gap, or both?

- This community-led approach directly shaped our dataset. We recorded 40 diverse speakers—including women with disabilities—covering two key Pulaar dialects: Fouta Toro and Fouta Djalion. We worked with two translators and an expert linguist for language accuracy, and engaged two maternal health experts to validate health terms across both dialects. This ensured marginalized voices and regional variations were accurately represented from day one.


## Stage 3 & 4. Model Benchmarking and Fine-Tuning

Open-source ASR models were benchmarked on Pulaar health-related speech using Word Error Rate (WER) and Character Error Rate (CER). A selected model was fine-tuned on the collected data to improve recognition of domain-specific vocabulary and dialectal variation.

#### Key Results

| Evaluation measure | Result |
|---|---:|
| Baseline WER | Approximately 43% |
| Fine-tuned WER | Approximately 28% |

#### GEDI-Informed Evaluation

The evaluation considered potential recognition challenges across:

- **Gender:** Differences in transcription performance between speaker groups.
- **Dialect:** Recognition of regional expressions and linguistic variation.
- **Accent:** Differences in pronunciation and speaking patterns.
- **Code-switching:** Speech that mixes Pulaar with French or other languages.




### 5. Testing Beyond the Laboratory


We tested the system not only for speech-to-text accuracy, but also for how well it serves underserved Pulaar-speaking women in real-world healthcare contexts.

#### Testing approach under GEDI Lens

- **Real-world conditions**: Tested in both quiet and noisy environments.
- **Speech robustness**: Spontaneous speech participants created their own narratives and spoke naturally to test real-time transcription in realistic situations and predefined scenarios, participants followed structured scenarios designed to assess how accuracy varied across accent, region, gender, and dialect.
- **Usability & accessibility**: Evaluated ease of use, comprehension, and overall user experience using likert scale 
- **Cultural & linguistic fit** : Examined dialects, local expressions, and lived contexts.


Scoring Principle: Each aspect is rated from 1 to 5 using clear criteria to ensure consistent evaluation across participants and testing sessions. (1 = Very Poor → 5 = Excellent)

#### Participant

We worked with ~10 users from diverse speaker groups and dialects across Senegal. Most participants were women, with a focus on underserved communities. The testing also included 2 community health workers and 2 linguists to assess health terminology, local expressions, and dialectal accuracy.


#### Two layers of evaluation

| Layer | What it tells you | Where |
|-------|-------------------|-------|
| **Automatic benchmark** | WER / CER against reference transcriptions, overall and by gender and dialect | this page + `evaluate_asr.py` |
| **Stage 6: User testing & validation** | Whether the transcriptions are accurate, meaningful, usable and faithful to the dialect for real users (GEDI lens) | [`user_testing/`](user_testing/README.md) |

#### results at a glance

| Dimension | Score |
|-----------|-------|
| ASR quality | 4 / 5 |
| Dialect and linguistic fidelity | 3 / 5 |
| Platform usability | 5 / 5 |
| **Overall** | **4 / 5** |

Usability and ASR quality were rated well. The main gap is **dialect and linguistic
diversity**, which is why the automatic benchmark reports results by dialect.
Details, participants and protocol: [`user_testing/README.md`](user_testing/README.md).

These results indicate improved transcription performance after fine-tuning. User testing also highlighted persistent challenges with dialect-specific expressions, accents, and spelling conventions.


## 6. Repository Structure

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

 # Part II:  Evaluate your own model on maternalth dataset 

A single script that evaluates a fine-tuned **Wav2Vec2-CTC** speech recognition
model on Pulaar test data and reports **WER** (Word Error Rate) and **CER**
(Character Error Rate).

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

## Quick start

```bash
git clone <https://github.com/YUX-Cultural-AI-Lab/pulaar-health-asr-gedi>
cd pulaar-asr-eval
pip install -r requirements.txt

# Evaluate on maternal health dataset covering 2langages over 200 samples
 python evaluate_asr.py --model <your-username/your-pulaar-model> --dataset pularheath --num-samples 200  # if your model is on huggingface
 python evaluate_asr.py --model ./path/to/your/model # or a local folder:
```

Your model must be a **Wav2Vec2 model with a CTC head** (`Wav2Vec2ForCTC`),
stored in a local folder or on the Hugging Face Hub.

The dataset is downloaded automatically. If something is private, run
`huggingface-cli login` first (or set `HF_TOKEN`).

## Options

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

**Step by step**

1. **Data loading.** Audio is read as raw WAV bytes (`decode=False`) and decoded
   with SoundFile, which avoids the TorchCodec requirement of recent `datasets`
   versions. Gender and dialect labels are read from the dataset columns.
   
3. **Audio preparation.** Stereo is averaged to mono and the signal is resampled to
   the rate the model was trained on (16 kHz for Wav2Vec2).
   
5. **Feature extraction.** `Wav2Vec2Processor` normalises the waveform into model inputs.
6. **Inference.** The model outputs, for every ~20 ms audio frame, a probability
   distribution over the character vocabulary (logits).
    **Inference.** The model gives, for every ~20 ms frame, scores over the
   character vocabulary.
   
8. **Decoding.** Greedy CTC decoding: take the most likely token per frame, merge
   consecutive duplicates, drop the blank token. No language model is used.
9. **Scoring.**
   - **WER** = (substitutions + deletions + insertions) / number of reference words.
     It can exceed 100% when the model outputs many wrong extra words.
   - **CER** = same idea at character level. It is usually more informative for a
     language with a young standard orthography like Pulaar, where a one-letter
     spelling difference (`pulaagu` vs `puraagu`) counts as a whole wrong word in WER.

## Outputs

- `results/results.json` – model, dataset, number of samples, WER, CER, runtime.
- `results/predictions.csv` – reference, prediction and per-sample WER for every
  example, handy for error analysis.

## Our results Results on different models

| Dataset | Samples | WER |
|---------|---------|-----|
| `guizme/Pulaar` (test) | 100 | --xx-- |
| `google/fleurs` `ff_sn` (test) | 100 | _xx_ |


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

