# Vaani (वाणी) 🎙️
### A Voice Dispatcher That Confirms Before It Acts
**Built for the Great Indian AI Internship Challenge by Gnani AI**  
**Target Award Category:** *Noisy Telephonic Audio*  
**Submission Deadline:** November 10, 2026 (Target: November 9)

---

## 1. The Core Idea

Truck drivers and cold-chain ground crews operate in deafening cabin environments (diesel engine rumble, highway wind, horn reverberation) where typing on a mobile screen at 70 km/h is impossible and speech recognition inevitably makes acoustic mistakes (confusing *18°C* with *80°C* or *14 km* with *40 km*). A single misread number can spoil an entire refrigerated consignment or dispatch roadside mechanics to the wrong highway milestone.

**Vaani's Claim:**  
> **Read-back confirmation catches speech-recognition errors that would otherwise cause a wrong dispatch. The safety gate is ordinary code, not the language model.**

The language model proposes structured details. The Python state machine validates them against strict physical constraints, reads them aloud in natural Hinglish to the driver, and **only the driver's affirmative confirmation unlocks the dispatch action**.

---

## 2. Benchmark Proof (The Evidence)

Our empirical benchmark evaluates **100 audio clips** across 4 acoustic conditions:
- **Clean:** Studio quality 16 kHz Hinglish recordings.
- **Telephony 8 kHz:** Resampled to 8 kHz with a digital 4th-order Butterworth bandpass filter (300–3400 Hz passband).
- **Telephony + 10 dB SNR:** 8 kHz telephony mixed with Indian truck diesel engine and tyre noise at 10 dB Signal-to-Noise Ratio.
- **Telephony + 5 dB SNR:** 8 kHz telephony mixed with severe cabin rumble and horn reflections at 5 dB SNR.

### Official Scorecard Track 1: Dialogue Controller & Safety Gate Stress-Testing (Acoustic Noise Simulation & Calibrated Traps)

> **Evaluation Protocol & Purpose:**  
> Track 1 isolates the finite state machine and dialogue safety layer from ASR variation by using calibrated acoustic degradation and scripted driver replies.
> Audio was synthesized using `gTTS` and mixed with calibrated synthetic Indian truck diesel engine and cabin noise.
> In this track, specific phonetic confusion traps (18↔80, 14↔40, negative polarity loss) are injected into the input to stress-test the state machine's interception capability.
> Driver replies model human factors: in loud cabin noise, drivers have an assumed reflexive error probability ($0\%$ in clean up to $12\%$ in severe noise).
> The scorecard below was generated directly by `benchmark/score.py` using `benchmark/results/raw_benchmark_results.csv`:

| Acoustic Condition | N | Source Engine | WER (%) | Raw Slot Acc (%) | Total Errors | Errors Caught by Read-Back | Silent Failures | Final Dispatch Accuracy (%) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean (16 kHz)** | 25 | Acoustic Simulation (Calibrated SNR) | 0.00% | 96.0% | 1 | **1 of 1 (100.0%)** | **0 (0.0%)** | **100.0%** |
| **Telephony 8 kHz** | 25 | Acoustic Simulation (Calibrated SNR) | 0.00% | 96.0% | 1 | **1 of 1 (100.0%)** | **0 (0.0%)** | **100.0%** |
| **Telephony + 10 dB SNR** | 25 | Acoustic Simulation (Calibrated SNR) | 1.72% | 72.0% | 7 | **6 of 7 (85.7%)** | **1 (4.0%)** | **96.0%** |
| **Telephony + 5 dB SNR** | 25 | Acoustic Simulation (Calibrated SNR) | 3.16% | 64.0% | 9 | **9 of 9 (100.0%)** | **0 (0.0%)** | **100.0%** |

---

### Official Scorecard Track 2: Empirical Model Evaluation (Held-Out Test Set N = 7 Utterances, 28 Clips)

To measure actual acoustic generalization without data leakage, `core/train_asr.py`, `core/train_denoiser.py`, and `core/train_nlu.py` quarantine 7 held-out utterances (`utt_03, utt_07, utt_11, utt_15, utt_19, utt_22, utt_25`). 
The table below reports real Word Error Rate (WER) measured by `benchmark/evaluate_trained_models.py` using the trained Whisper-LoRA model with and without the 8 kHz neural denoiser (saved in `benchmark/results/model_evaluation_metrics.csv`):

| Condition | In-Domain (N=18) Raw WER | In-Domain (N=18) Denoised WER | Held-Out (N=7) Raw WER | Held-Out (N=7) Denoised WER | Overall (N=25) Raw WER | Overall (N=25) Denoised WER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean (16 kHz)** | **1.58%** | 1.98% | **21.05%** | 23.16% | **6.90%** | 7.76% |
| **Telephony 8 kHz** | **1.58%** | 2.37% | **31.58%** | 28.42% | **9.77%** | 9.48% |
| **Telephony + 10 dB SNR** | **2.77%** | 3.16% | **32.63%** | 36.84% | **10.92%** | 12.36% |
| **Telephony + 5 dB SNR** | **4.74%** | 13.83% | **40.00%** | 43.16% | **14.37%** | 21.84% |

#### Key Empirical Diagnostics & Insights:
1. **Script Alignment Diagnostic:**  
   In initial evaluation runs, setting `language="hi"` forced Whisper's multilingual head to decode into Devanagari Hindi script (`भाया रीफर का...`), which scored an artificial ~100% substitution WER against Roman Hinglish references (`Bhaiya reefer ka...`). Removing language coercion aligns Whisper with Roman Hinglish fine-tuning, bringing In-Domain Raw WER down to **1.58%** on clean and **4.74%** at 5 dB SNR.
2. **Denoiser Ablation:**  
   A spectral denoiser trained on small local audio data introduces phase distortion and spectral smearing at low SNRs (at 5 dB SNR in-domain WER rises from 4.74% raw to 13.83% denoised). Consequently, **direct raw audio into Whisper is the recommended default pipeline**.
3. **Held-Out NLU Accuracy:**  
   Evaluating the deterministic regex extractor on the 7 held-out utterances yields **94.7% slot accuracy** (18 of 19 mandatory slots extracted correctly across cold-chain landmarks and temperatures), confirming that deterministic code provides robust out-of-distribution guardrails.

---

### Illustrative Driver Attention Sensitivity Sweep (0% to 20% Reflexive Error Rate)

> **Methodological Note:** The table below is an illustrative parametric sensitivity sweep ($E_{\text{silent}} = \text{Errors} \times \text{Assumed Rate}$), demonstrating the mathematical boundary conditions of read-back confirmation under varying driver fatigue and distractibility, rather than a physical measurement of driver psychology.

| Assumed Driver Error Rate | Clean (16 kHz) Dispatch Acc | Telephony 8 kHz Dispatch Acc | 10 dB SNR Dispatch Acc | 5 dB SNR Dispatch Acc |
| :---: | :---: | :---: | :---: | :---: |
| **0% (Attentive Driver)** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **5% (Minor Distraction)** | **99.8%** | **99.8%** | **98.6%** | **98.2%** |
| **10% (Moderate Noise)** | **99.6%** | **99.6%** | **97.2%** | **96.4%** |
| **15% (Heavy Fatigue)** | **99.4%** | **99.4%** | **95.8%** | **94.6%** |
| **20% (Severe Noise & Fatigue)** | **99.2%** | **99.2%** | **94.4%** | **92.8%** |

*Raw data logged in `benchmark/results/driver_sensitivity_sweep.csv`.*

---

### Train vs. Held-Out Evaluation Protocol
To avoid data contamination and memorization on small corpora:
- **Training Set (N = 18 utterances x 4 variants = 72 clips):** `core/train_asr.py` and `core/train_nlu.py` strictly train only on the 18 training utterances.
- **Held-Out Test Set (N = 7 utterances x 4 variants = 28 clips):** Utterances `utt_03, utt_07, utt_11, utt_15, utt_19, utt_22, utt_25` are strictly quarantined and never seen during gradient updates. Evaluated separately in `benchmark/evaluate_trained_models.py`.
- **Dynamic Evon Cache Invalidation:** Evon extraction cache keys incorporate the MD5 hash of active phonetic rules (`get_rules_version_hash()`). Stored cache entries are automatically invalidated whenever continuous learning synthesizes new disambiguation rules.

---

## 3. System Architecture & Dual ASR Execution

Vaani supports both **Cloud STT (Gnani Prisma)** and **On-Device Local GPU STT (Fine-Tuned Whisper-LoRA)** with a neural 8 kHz front-end denoiser:

```
[Driver Hinglish Speech] 
          │
          ▼
┌────────────────────────────────────────┐
│ [OPTIONAL] 8 kHz Neural Denoiser       │ (Conv-BiGRU spectral mask: suppresses diesel rumble)
│ Checkpoint: models/denoiser_8khz.pt    │
└─────────────────┬──────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────┐
│ DUAL ASR ENGINE:                       │
│ - Fine-Tuned Whisper-LoRA (Local GPU)  │ (Evaluated on held-out N=7 unseen clips; sub-second latency)
│ - OR Gnani Prisma v3.0 (Cloud API)     │
└─────────────────┬──────────────────────┘
                  │ Spoken Transcript
                  ▼
[Gnani Evon / Few-Shot Engine] ── (Extracts slots: incident, temp, landmark, distance)
                  │
                  ▼
[Pydantic Validation] ─────────── (Rejects invalid bounds: temp outside -40°C to +50°C)
                  │
                  ▼
[Dialogue Controller] ─────────── (Plain Python FSM: LISTENING -> EXTRACTING -> CONFIRMING)
                  │
                  ▼
[Gnani Timbre TTS] ────────────── (Spoken Hinglish: "Temperature 18 degree, toll plaza se 42 km aage. Sahi hai?")
                  │
             ┌────┴───────────────────────────┐
             ▼                                ▼
       [Driver: "Nahi, 18 degree"]      [Driver: "Haan, sahi hai"]
             │                                │
             ▼                                ▼
       [CORRECTING state]               [ACTING state] ─── (ONLY state allowed to call SQLite)
       (Max 2 retries, else escalate)        │
                                              ▼
                                [Simulated Emergency Dispatch]
```

### The Most Important Design Rule
The dispatch function checks:
```python
if self.state != DialogueState.ACTING:
    raise PermissionError("SAFETY GATE VIOLATION: Dispatch action rejected! State is not ACTING.")
```
`ACTING` can strictly only be reached when a positive confirmation reply is received during `CONFIRMING`.

---

## 4. Repository Structure

```
VoiceAI/
├── README.md                      # Complete project documentation & limitations
├── requirements.txt               # Dependencies
├── app.py                         # Root entrypoint
├── app/
│   └── dashboard.py               # Streamlit Telematics Console & Benchmark UI
├── core/
│   ├── schemas.py                 # Pydantic schemas for extraction & validation
│   ├── state_machine.py           # Finite State Machine & Safety Gate
│   └── tools.py                   # Simulated fleet operations (lookup, dispatch, ETA)
├── gnani/
│   ├── client.py                  # Thin wrappers for Prisma, Evon, Timbre + Cache
│   └── __init__.py
├── benchmark/
│   ├── utterances.csv             # 25 Hinglish test cases with acoustic traps
│   ├── make_noisy.py              # Telephony 8 kHz filter & SNR noise mixing
│   ├── run_benchmark.py           # 100-clip execution engine with caching
│   ├── score.py                   # jiwer WER and slot accuracy scorer
│   ├── audio/                     # 100 synthesized and degraded WAV clips
│   └── results/
│       ├── raw_benchmark_results.csv
│       ├── summary_metrics.csv
│       └── benchmark_cache.json
├── data/
│   ├── fleet.sqlite               # Simulated fleet database (vehicles, tickets, ETAs)
│   ├── training_ledger.json       # Persistent data usage tracker (GBs, hours, utterances)
│   └── training_ledger.csv        # Tabular training audit export
├── models/
│   ├── whisper_hinglish_lora/     # Fine-tuned Whisper-LoRA adapters
│   ├── denoiser_8khz.pt           # 8 kHz Telephony Neural Denoiser
│   └── nlu_slot_extractor/        # Flan-T5 slot extraction adapter
└── tests/
    └── test_state_machine.py      # Pytest suite verifying the safety gate
```

---

## 5. Chunked Training & Data Usage Ledger

To train on massive Indian speech corpora (Gram Vaani ~1,100 hrs, FLEURS, agarwalayushi/hinglish ~2,200 hrs) without filling local disk space, Vaani implements a **Chunk -> Train -> Auto-Purge** loop:
1. **Bounded Staging:** Streams audio in controlled chunks (up to target GB / samples) to a temporary staging folder.
2. **GPU Optimization:** Fine-tunes Whisper-LoRA adapters using PyTorch on CUDA.
3. **Instant Auto-Purge:** Immediately deletes the raw audio chunk from disk upon gradient completion.
4. **Persistent Auditing:** Logs exact size in GB, spoken audio hours, and loss in `data/training_ledger.json` and in the Streamlit UI dashboard.

```bash
# Run a chunked training cycle (downloads chunk, trains, deletes raw audio, updates ledger):
python core/chunked_trainer.py --dataset dianavdavidson/gram-vaani-preprocessed --chunk_id chunk_01_gramvaani --samples 200

# View the real-time data usage ledger:
python core/chunked_trainer.py --summary
```

---

## 6. What Is Real vs. What Is Simulated

In adherence to Section 16 (Claims Policy):
- **Real:**
  - Gnani API integration architecture (Prisma STT, Evon LLM, Timbre TTS).
  - Fine-Tuned Whisper-LoRA and neural 8 kHz front-end denoiser running on GPU/CPU.
  - Telephony DSP pipeline: Exact 8 kHz resampling, 300–3400 Hz Butterworth bandpass filtering, calibrated SNR power calculation ($10 \cdot \log_{10}(P_{signal} / P_{noise})$).
  - Word Error Rate scoring calculated with `jiwer`.
  - Finite State Machine safety enforcement and Pydantic schema validation.
  - Disk caching preventing redundant API cost.
- **Simulated:**
  - Fleet vehicles, GPS coordinates, and dispatch tickets (`data/fleet.sqlite` is explicitly labeled "Simulated fleet data").
  - Background noise mixed into telephony clips (reproducible mathematical noise model of diesel engine harmonics, pink noise road friction, and horn reflections).
  - Driver replies during benchmark automation (calibrated human factor simulation with reflexive confirmation failure modes over harsh noise).

---

## 7. How to Run

### 1. Configure Environment & Dependencies
```bash
python -m pip install -r requirements.txt
# Copy example configuration and add your rotated GNANI_API_KEY from https://app.gnani.ai
cp .env.example .env
```

### 2. Run Comprehensive Safety Gate Pytest Suite
```bash
python -m pytest tests/ -v
```

### 3. Run Benchmark Harness
```bash
# Run benchmark runner across all 100 clips
python benchmark/run_benchmark.py

# Compute WER, Slot Accuracy, and Read-back Interception rates
python benchmark/score.py
```

### 4. Evaluate Held-Out Generalization
```bash
python benchmark/evaluate_trained_models.py
```

### 5. Launch Streamlit Telematics Dashboard
```bash
streamlit run app/dashboard.py
```

---

## 8. Limitations Stated Honestly

1. **Simulated Telemetry:** Fleet data, tickets, and driver replies in automated tests are simulated.
2. **Benchmark Scope:** The benchmark consists of 25 utterances (100 total degraded clips) synthesized using gTTS with calibrated synthetic diesel noise. While targeted at realistic acoustic confusions (18 vs 80, 14 vs 40, minus polarity), it is a focused evaluation corpus rather than an exhaustive national field study.
3. **Turn Latency:** Read-back confirmation adds 3–5 seconds to each driver call. In high-stakes cold-chain logistics, this latency is an intentional trade-off to prevent hours of wasted dispatch.
4. **Driver Fallibility:** A fatigued driver could conceivably confirm a wrong read-back by mistake; a production system requires secondary verification for high-value cargo.
