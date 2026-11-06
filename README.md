# FastSpeech 2 TTS: Multi-Lingual Speech Synthesis (English & Telugu)

An end-to-end Text-to-Speech (TTS) engine executing high-fidelity neural speech generation for both Indian-accented English (using the Svarah corpus) and Telugu (using the IndicTTS Telugu corpus). This project implements FastSpeech 2 (Ren et al., 2021) adapted from the ming024 baseline to support linguistic front-ends and pitch/energy/duration variance adaptors tailored for regional languages.

Detailed experimental analyses, training loss convergence curves, and acoustic evaluation remarks are documented in **Report/Technical_Report.pdf**.

---

## 📊 Empirical Metrics & System Overview

| Parameter | English Pipeline (Svarah) | Telugu Pipeline (IndicTTS) |
| :--- | :--- | :--- |
| **Dataset Capacity** | 5.05h Train / 1.05h Val (3,590 / 790 clips) | 5.04h Train / 1.02h Val (2,398 / 485 clips) |
| **Speaker Diversity** | 115 speakers (~2 mins/speaker) | 2 speakers (2.5h each) |
| **Training Steps** | 60,000 steps (~2 hours) | 60,000 steps (~2.5 hours) |
| **Optimal Loss (Step)**| 2.614 (at step 13k) | 1.960 (at step 32k) |
| **Inference Weights** | step 60,000 | step 32,000 (Best Validation checkpoint) |
| **ASR Quality (WER/CER)**| Synthetic: 18.2% vs. Ground-Truth: 5.7% | Synthetic: 15.8% vs. Ground-Truth: 16.2% |

---

## 🗂 Project Architecture

```text
project/
├── fastspeech2/            # Shared neural synthesis & processing utilities
│   ├── model/              # Transformer core, length regulators, and loss modules
│   ├── text/english.py     # Custom POS-tagged English text normalizer & heteronym analyzer
│   ├── text/telugu.py      # Akshara-level Telugu normalizer & rule-based G2P engine
│   ├── preprocessor.py     # Feature extraction (spectrogram, pitch, energy, duration)
│   └── asr_eval.py         # Intelligibility metrics generation using Whisper
├── English/                # Training & local inference entrypoints for Svarah
├── Telugu/                 # Training & local inference entrypoints for IndicTTS
├── Results/                # Generated audio outputs and pitch/energy/duration graphs
├── Report/                 # Submission-ready PDF research report
└── README.md
```

---

## 🛠 Quickstart Guide

### 1. Dependency Installation
```bash
# Create and isolate the main environment
conda create -n tts-engine python=3.10 -y
conda activate tts-engine

# Install CUDA-compatible PyTorch wheels and base requirements
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 2. Run Inference (Pretrained Models)
To synthesize the required assignment sentences and output custom files directly to the `Results/` folder:
```bash
# Synthesize English test set
cd English
python inference.py

# Synthesize Telugu test set
cd ../Telugu
python inference.py
```
To test custom inputs with specific voice overrides:
```bash
python inference.py --text "Your custom sentence here" --id custom_test_run
```

---

## 🔬 Methodology & Implementation Enhancements
- **Acoustic Front-ends:** Designed and integrated dedicated phonetic front-ends. The English module handles heteronyms via POS tagging using NLTK; the Telugu module maps Aksharas to 46 customized phone classes.
- **Robust Feature Preprocessing:** Parallelized acoustic feature extraction, customized training/validation split to prevent leakages, and bounded length sequences to 1,300 frames to stabilize variance predictions.
- **Optimized Training:** Consolidated hyperparameters into a unified `config.yaml` to govern model training, utilizing lightweight FP16 checkpoint architectures.

---

## 🎓 Credits & License
- Core neural architectures and vocoders are modified from `ming024/FastSpeech2` and `HiFi-GAN` (Kong et al.).
- Dataset access provided by SPRINGLab (Svarah) and the IndicTTS Consortium (Telugu).
