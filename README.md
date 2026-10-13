# Pulaar Health ASR: GEDI-Informed Speech Recognition

**Developing and evaluating inclusive speech recognition for Pulaar maternal and reproductive health communication.**

## Overview

Automatic Speech Recognition (ASR) can help make health communication more accessible by converting spoken language into text. However, ASR systems often struggle with low-resource African languages, dialectal variation, local health terminology, and code-switching.

This project explores the development and evaluation of ASR for Pulaar maternal and reproductive health communication, with a focus on **Gender, Equity, Diversity, and Inclusion (GEDI)**.

The project was conducted under the **GRAIN EmpowerHerAI initiative**, with a focus on representing diverse speakers, capturing regional linguistic variation, and evaluating the technology beyond overall transcription accuracy.

## Objectives

- Adapt open-source ASR models to Pulaar maternal and reproductive health speech.
- Evaluate speech recognition across the Fuuta Tooro and Fuuta Jallon dialects.
- Examine potential disparities related to gender, dialect, accent, and code-switching.
- Integrate linguistic expertise and community feedback into data validation and system evaluation.
- Assess usability and accessibility in realistic use conditions.

## Methodology

The project followed a GEDI-informed approach throughout the ASR development and evaluation process.

### 1. Data Collection and Validation

We collected maternal and reproductive health utterances from **40 Pulaar speakers**, covering the Fuuta Tooro and Fuuta Jallon dialects and including women with disabilities.

Data validation involved linguistic expertise, translation review, and maternal-health specialists to improve the accuracy and contextual relevance of the collected utterances.

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

