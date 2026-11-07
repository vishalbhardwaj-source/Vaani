# Vaani Empirical Benchmark Methodology & Experimental Protocol

## 1. Dual-Track Benchmark Architecture

To avoid misleading claims, Vaani evaluates speech recognition and dialogue action on two clearly decoupled tracks:

- **Track 1: Dialogue-Gate Stress Test with Injected ASR Errors (Synthetic / Unit Stress)**  
  Isolates the state machine and confirmation gate from acoustic model variance. Injects calibrated phonetic traps (18↔80, 14↔40, sign drops) and models driver reflexive confirmation (0% to 20%) to verify that code catches injected errors.
- **Track 2: Empirical Acoustic Generalization (Real Held-Out Audio & Real Phone Cohort)**  
  Measures true acoustic performance on 28 held-out audio clips (7 unseen test utterances across 4 acoustic conditions) and 40 mobile phone clips recorded with 8 commercial truck drivers (N=80 numeric slots).

---

## 2. Multi-Seed Generalization & Denoiser Ablation Analysis

Ablation evaluation of the 8 kHz neural denoiser (`models/denoiser_8khz.pt`) across channel conditions:

| Acoustic Condition | Raw Whisper-LoRA WER (%) | Denoised WER (%) | Denoising Impact |
| :--- | :---: | :---: | :--- |
| **Clean (16 kHz)** | 21.05% | 22.11% | Negligible (-1.06 pp) |
| **Telephony 8 kHz** | 23.16% | 23.16% | Neutral (0.00 pp) |
| **Telephony 10 dB SNR** | 33.68% | 41.05% | **Hurts (+7.37 pp)** |
| **Telephony 5 dB SNR** | 42.11% | 53.68% | **Severely Hurts (+11.57 pp)** |

> **Diagnostic Finding:** At low SNR (<10 dB), neural phase-estimation and spectral masking introduce phase artifacts that degrade downstream transformer ASR token generation. The denoiser is therefore configured as an optional pre-filter, disabled by default in severe highway noise.

---

## 3. Parametric Driver Reflexive Agreement Sweep

In loud vehicle cabs, drivers occasionally say *"haan"* reflexively without listening attentively to readback details. We model this via Monte Carlo sensitivity sweep (0% to 20% driver reflexive error rate):

| Assumed Reflexive Rate | Total Injected Errors | Gate Gated (IoT) | Expected Driver Caught | Expected Silent Failures | Final Dispatch Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0%** | 29 | 12 | 17.0 | 0.0 | **100.0%** |
| **5%** | 29 | 12 | 16.15 | 0.85 | **99.1%** |
| **10%** | 29 | 12 | 15.30 | 1.70 | **98.3%** |
| **15%** | 29 | 12 | 14.45 | 2.55 | **97.4%** |
| **20%** | 29 | 12 | 13.60 | 3.40 | **96.6%** |
