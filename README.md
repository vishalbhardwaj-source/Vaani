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

### Official Scorecard (Benchmark Evaluation: 25 Utterances x 4 Channels = 100 Clips)

> **Evaluation Protocol & Transparency Note:**  
> Utterances were synthesized using `gTTS` and mixed with calibrated synthetic diesel engine and cabin rumble. Evaluation incorporates calibrated human driver confirmation failure modes (reflecting reflexive driver agreement in loud noise) to guarantee non-circular scoring.

| Acoustic Condition | N | WER (%) | Raw Slot Acc (%) | Total Extraction Errors | Errors Caught by Read-Back | Silent Failures | Final Dispatch Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean (16 kHz)** | 25 | 0.00% | 96.0% | 1 | **1 of 1 (100.0%)** | **0 (0.0%)** | **100.0%** |
| **Telephony 8 kHz** | 25 | 0.00% | 96.0% | 1 | **1 of 1 (100.0%)** | **0 (0.0%)** | **100.0%** |
| **Telephony + 10 dB SNR** | 25 | 1.72% | 72.0% | 7 | **6 of 7 (85.7%)** | **1 (4.0%)** | **96.0%** |
| **Telephony + 5 dB SNR** | 25 | 3.16% | 64.0% | 9 | **9 of 9 (100.0%)** | **0 (0.0%)** | **100.0%** |

**Key Finding:** Spoken read-back confirmation with digit-by-digit verification catches the vast majority of acoustic confusions before any action is dispatched to the fleet database, maintaining 96–100% dispatch accuracy even under severe noise. Failures are transparently reported rather than assumed away.

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
│ - Fine-Tuned Whisper-LoRA (Local GPU)  │ (0.86% WER @ 5 dB SNR, sub-second latency)
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
