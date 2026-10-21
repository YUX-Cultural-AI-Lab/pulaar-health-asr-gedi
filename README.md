

# Beyond Accuracy: Building Inclusive Speech Recognition for Pulaar Through a Gender and Dialect Lens

## Overview

ASR can improve maternal healthcare access, but limited data and linguistic diversity make it difficult for systems to accurately recognize African languages and dialects. Overall accuracy may also hide disparities across genders and regional varieties, such as Fuuta Toro and Fuuta Jalon Pulaar. Technical accuracy alone is not enough: inclusive AI requires active community participation to ensure systems reflect users’ lived experiences, priorities, and needs.

This study presents a Gender, Equity, Diversity, and Inclusion (GEDI) approach to developing and evaluating Pulaar ASR in multidialect languages(settings) in Africa

## Objectives

Integrated GEDI considerations across the AI lifecycle, from community engagement and data collection to model training, evaluation, deployment, and user testing
 
#### Repository Structure

```text
pulaar-health-asr-gedi/
├── Overview
├── Objectives
├── Methodology
├── GEDI integration over AI cycle
├── audios/
├── data/
│   └── pular_dataset.csv
├── evaluation/
│   ├── benchmark_asr.py
│   └── eval_asr.py
├── results/
│   ├── Eval_results.csv
│   ├── benchmark_results.csv
│   ├── Eval_results.json
│   └── benchmark_results.json
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```


## Methodology

We integrated GEDI considerations across the AI lifecycle, We collected speech data from 40 speakers across the Fuuta Tooro and Fuuta Dallon dialects, including women with disabilities, with linguistic and maternal-health experts supporting validation. We fine-tuned the ASR model and evaluated transcription performance using WER, alongside dialect fidelity and linguistic accuracy. Finally, user testing assessed transcription quality, dialect representation, and platform usability.

We also provide an evaluation script that enables researchers to assess their own ASR models on Pulaar maternal-health speech data and compare their performance against our state-of-the-art model.

### Stage 1 & 2 : Community Engagement & Data collection

- The initial phase focused on deeply understanding the target community, assessing local access to technology, and aligning our research objectives with real-world needs. A central question guided this stage: Are we addressing a genuine community priority, a technical research gap, or both?

- This community-led approach directly shaped our dataset. We recorded 40 diverse speakers—including women with disabilities—covering two key Pulaar dialects: Fouta Toro and Fouta Djalion. We worked with two translators and an expert linguist for language accuracy, and engaged two maternal health experts to validate health terms across both dialects and sociologists and social-behavioural experts with experience working in rural areas helped us understand community priorities and lived realities.


### Stage 3 & 4. Model Benchmarking and Fine-Tuning

Open-source ASR models were benchmarked on Pulaar health-related speech using Word Error Rate (WER) and Character Error Rate (CER). A selected model was fine-tuned on the collected data to improve recognition of domain-specific vocabulary and dialectal variation.

  
- #### Benchmarking on maternal health pular dataset

| ASR Models |Architecture | WER (Word Error rate)| CER (character Error rate)|
|---|---|---|---:|
|Wav2Vec2-Fula |Wav2Vec2|  43.81 %| 14.14 % |
| ASR-Africa-Fula |Wav2Vec2| 91.24% | 32.88 %|
| Whisper-Large-v3 |Whisper|  124.02 %| 61.38 %|

- #### Key Results after finetuning

| models | WER (Word Error rate)| CER (character Error rate)|
|---|---|---:|
|Wav2Vec2-Fula (Baseline ) | 43.81 %| 14.14 % |
| Wav2Vec2-Fula (finetuned ) | 28.81% | 7.97 %|

- #### GEDI-Informed Evaluation

The evaluation considered potential recognition challenges across:

- **Gender:** Differences in transcription performance between speaker groups.
- **Dialect:** Recognition of regional expressions and linguistic variation.
- **Accent:** Differences in pronunciation and speaking patterns.
- **Code-switching:** Speech that mixes Pulaar with French or other languages.

## 5. Testing Beyond the Laboratory

We tested the system not only for speech-to-text accuracy, but also for how well it serves underserved Pulaar-speaking women in real-world healthcare contexts.

### Testing approach under GEDI Lens

- **Automatic benchmark** : WER / CER against reference transcriptions, overall and by gendern <span style="color: #166534;"> `evaluate_asr.py` </span>
   - **Real-world conditions**: Tested in both quiet and noisy environments.
  - **Speech robustness**: Spontaneous and read speech

- **User testing & field validation** Assess how effectively the system serves community. 
   - **Usability & accessibility**: Evaluated ease of use, comprehension, and overall user experience using likert scale, accesibility, , trust, usefulness
   - **Cultural & linguistic fit** : Examined dialects, local expressions, language fidelity, and lived contexts.

<i> Scoring Principle: Each aspect is rated from 1 to 5 using clear criteria (1 = Very Poor → 5 = Excellent)<i>

### results at a glance

| Dimension | Score |
|-----------|-------|
| ASR quality | 4 / 5 |
| Dialect and linguistic fidelity | 3 / 5 |
| Platform usability | 5 / 5 |


Usability and ASR quality were rated well. The main gap is **dialect and linguistic diversity**, which is why the automatic benchmark reports results by dialect.
Details, participants and protocol:<span style="color: #166534;"> [`user_testing/README.md`](user_testing/README.md). </span>

These results indicate improved transcription performance after fine-tuning. User testing also highlighted persistent challenges with dialect-specific expressions, accents, and spelling conventions.


*This structure describes the proposed organization of the repository. Files should be added as the corresponding code and documentation become available.*

 ## Part II:  Evaluate your own model on maternal health dataset 

A single script that evaluates a fine-tuned **Wav2Vec2-CTC** speech recognition model on Pulaar test data and reports **WER** (Word Error Rate) and **CER** (Character Error Rate).

We provide:

1. **A test dataset** of 100 maternal-health audio samples with transcriptions, speaker **gender** and **dialect**: <DATASET_LINK>
2. **Our reference results**, obtained with our own model
   ([`YUXCulturalAILab/wav2vec_pular_maternalhealth`](https://huggingface.co/YUXCulturalAILab/wav2vec_pular_maternalhealth)),
   in [`results /`](results/).
3. **An evaluation script** so you can run *your* model on the same data and compare your scores with ours, overall and by gender and dialect.
4. **A user testing and validation protocol (Stage 6)** with our results from real users, including Community Health Workers and linguists: [`user_testing/`](user_testing/).

## Quick start

```bash
git clone <https://github.com/YUX-Cultural-AI-Lab/pulaar-health-asr-gedi>
cd pulaar-asr-eval
pip install -r requirements.txt

#  Reproducing our benchmark

 python evaluation/benchmark_asr.py 
 --model "YUXCulturalAILab/wav2vec_pular_maternalhealth"
 --dataset data/pular_dataset.csv 
 --num-samples 100

# Evaluate your own model on pular dataset maternal and reproductive health

 python evaluate_asr.py --model <your-username/your-model>  # if your model is on huggingface
 --dataset data/pular_dataset.csv 
 --num-samples 100  
 python evaluate_asr.py --model ./path/to/your/model # if your model is in local folder:
```

The dataset is downloaded automatically. If something is private, run
`huggingface-cli login` first (or set `HF_TOKEN`).

## Other options

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
   - **WER** = (substitutions + deletions + insertions) / number of reference words. It can exceed 100% when the model outputs many wrong extra words.
   
   - **CER** = same idea at character level. It is usually more informative for a language with a young standard orthography like Pulaar, where a one-letter spelling difference (`pulaagu` vs `puraagu`) counts as a whole wrong word in WER.

#### Outputs

- `results/benchmark_results.json` – model, dataset, number of samples, WER, CER, runtime.
- `results/eval_result.csv` – reference, prediction and per-sample WER for every
  example, handy for error analysis.

#### What you get

The script reports the overall WER and CER for your model, along with evaluation results grouped by dialect and gender

```
 Evaluation on your dataset
----------------------------------------------------------------------
    model                        language      samples    wer    cer
----------------------------------------------------------------------
your-username/your-pulaar-model  Pulaar          100    xx.xx  xx.xx
----------------------------------------------------------------------


 Reproducing our benchmark 
-----------------------------------------------------------
Model  architecture               samples    wer    cer            
-----------------------------------------------------------
Wav2Vec2-Fula                       100     43.81  14.14  
ASR-Africa-Fula                     100     91.24  32.88  
Whisper-Large-v3                    100     124.02 61.38  
your-username/your-pulaar-model     100     xx.xx  xx.xx  
-----------------------------------------------------------
```
and saves, in `results/`:



| File | Content |
|------|---------|
| `results.json` | Overall + per-gender + per-dialect WER/CER |
| `breakdown.csv` | Same breakdown as a table |
| `predictions.csv` | Reference vs prediction for every sample, with gender and dialect |


#### Notes

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

**01 — Expand Speech Data Collection**: Extend Pulaar ASR coverage beyond Fuuta Toro and Fuuta Jalon (Senegal) to additional Fulfulde varieties, including those spoken in Nigeria, prioritizing underserved speakers, dialect-specific health terminology, and code-switching.

**02 — Two-Way Voice Healthcare**: Develop a voice-based healthcare solution that transcribes patient–provider interactions, translates between Pulaar and healthcare providers’ languages, and delivers doctors’ responses in spoken Pulaar to improve access to maternal healthcare.
## Acknowledgements

This work was conducted under the GRAIN EmpowerHerAI initiative. The project involved collaboration with linguistic and maternal-health experts and contributors to Pulaar data collection and validation.

Partner acknowledgements and institutional affiliations should be finalized in accordance with project agreements.

## Citation

To come ...

## License

The code and documentation in this repository are licensed under the Apache License 2.0. See the [LICENSE](LICENSE.md) file for details.

Dataset and model licenses may differ from the repository's software license. Please refer to the relevant dataset and model documentation for their respective terms of use.

