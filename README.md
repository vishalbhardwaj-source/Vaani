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

### Official Scorecard Track 1: Dialogue Controller & Safety Gate Stress-Testing (Calibrated Acoustic Noise Traps)

> **Evaluation Protocol & Purpose:**  
> Track 1 isolates the finite state machine and dialogue safety layer from ASR variation by using calibrated acoustic degradation and scripted driver replies.
> Audio was synthesized using `gTTS` and mixed with calibrated synthetic Indian truck diesel engine and cabin noise.
> **Fleet Telemetry Stand-in (Oracle Mapping):** Each utterance is mapped to its matching fleet cargo vehicle (`HR-55-TC-1102` for sub-zero ice cream target −22°C, `DL-01-AA-4092` for dairy target +4°C, `GJ-01-BX-7201` for fresh produce target +6°C). In production, the cargo target comes from fleet data, independent of what the driver says. The telematics plausibility policy challenges any reported reading deviating >25°C from target cargo limits.
> Specific phonetic confusion traps (18↔80, 14↔40, negative polarity loss) are injected into the input to stress-test the state machine's interception capability.
> Driver replies model human factors: in loud cabin noise, drivers have an assumed reflexive error probability (0% in clean up to 12% in severe noise).
> **Scorecard Outcome Categorization:** A blocked clip is safe (preventing false dispatch) but is not a successful dispatch, so it is tracked as its own distinct category. Total Errors = Driver Corrected Errors + Gate Blocked Anomalies + Silent Failures. Total Clips N = Successful Dispatches + Blocked Anomalies + Silent Failures.
> **Slot Metrics Focus:** Track 1 focuses strictly on slot extraction and dialogue controller interception. (Synthetic simulation WER is excluded from Track 1 because acoustic recognition accuracy is evaluated on real models in Track 2).
> The scorecard below was generated directly by `benchmark/score.py` using `benchmark/results/raw_benchmark_results.csv`:

| Condition | Sample Size (N) | Raw Slot Acc (%) | Raw Slot Error Rate (%) | Total ASR Errors | Driver Corrected Errors | Gate Blocked Anomalies (Safe) | Silent Failures | Successful Dispatches (%) | Blocked Dispatch Withheld (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| clean | 25 | 96.0 | 4.0 | 1 | 1 | 0 | 0 (0.0%) | 25 of 25 (100.0%) | 0 of 25 (0.0%) |
| telephony_8khz | 25 | 96.0 | 4.0 | 1 | 1 | 0 | 0 (0.0%) | 25 of 25 (100.0%) | 0 of 25 (0.0%) |
| telephony_10db | 25 | 72.0 | 28.0 | 7 | 6 | 0 | 1 (4.0%) | 24 of 25 (96.0%) | 0 of 25 (0.0%) |
| telephony_5db | 25 | 64.0 | 36.0 | 9 | 9 | 0 | 0 (0.0%) | 25 of 25 (100.0%) | 0 of 25 (0.0%) |

*Note: In the 10 dB SNR condition, the single silent failure (4.0%) represents a single clip where the assumed 6% driver reflexive agreement allowed a minor unflagged extraction error to slip through.*

---

### Adversarial Telematics Gate Benchmark (Plausible Anomalies Stress-Test)

To empirically demonstrate that the telematics gate intercepts plausible out-of-range readings that slip past ASR, 10 adversarial scenarios were evaluated across different vehicle cargo classes (`benchmark/results/adversarial_gate_results.csv`).
In every scenario, the reported temperature is physically possible for reefers (−40°C to +50°C), but diverges widely (>25°C) from the fleet telematics target:
- **Test Mode A (Reflexive Agreement):** The driver mishears or reflexively replies with `"haan"`, `"haan bhaiya sahi hai"`, or bare `"confirm"`. **Result: 10 of 10 dispatches strictly BLOCKED (100% Interception, 0 false dispatches)**.
- **Test Mode B (Verified Number Repeat):** The driver explicitly repeats the number (`"X degree confirm"`). **Result: 10 of 10 dispatches verified and permitted (100% Success)**.

| Scenario ID | Vehicle ID | Cargo Target (°C) | Reported Temp (°C) | Target Divergence (°C) | Reflexive Reply (haan) Blocked | Verified Repeat Permitted |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| adv_01 | HR-55-TC-1102 | -22.0 | 12.0 | 34.0 | True | True |
| adv_02 | HR-55-TC-1102 | -22.0 | 18.0 | 40.0 | True | True |
| adv_03 | HR-55-TC-1102 | -22.0 | 25.0 | 47.0 | True | True |
| adv_04 | HR-55-TC-1102 | -22.0 | 35.0 | 57.0 | True | True |
| adv_05 | DL-01-AA-4092 | 4.0 | -24.0 | 28.0 | True | True |
| adv_06 | DL-01-AA-4092 | 4.0 | 32.0 | 28.0 | True | True |
| adv_07 | DL-01-AA-4092 | 4.0 | 40.0 | 36.0 | True | True |
| adv_08 | GJ-01-BX-7201 | 6.0 | -22.0 | 28.0 | True | True |
| adv_09 | GJ-01-BX-7201 | 6.0 | 35.0 | 29.0 | True | True |
| adv_10 | GJ-01-BX-7201 | 6.0 | 42.0 | 36.0 | True | True |

---

### Official Scorecard Track 2: Empirical Model Generalization (Held-Out Test Set N = 7 Utterances, 28 Clips)

To measure actual acoustic generalization without data leakage, `core/train_asr.py`, `core/train_denoiser.py`, and `core/train_nlu.py` quarantine 7 held-out utterances (`utt_03, utt_07, utt_11, utt_15, utt_19, utt_22, utt_25`).
The table below reports real Word Error Rate (WER) and **Number Survival Accuracy** (whether temperatures and distances survive transcription) measured by `benchmark/evaluate_trained_models.py` using the trained Whisper-LoRA model with and without the 8 kHz neural denoiser (saved in `benchmark/results/model_evaluation_metrics.csv`):

> **Sample Size & Retrain Variance Notice:** $N=7$ held-out utterances (28 audio clips across 4 channel variants; 10 total expected numeric slots) is a compact benchmark set. Observed ranges across retrains with prefix tokenization and hyperparameter tuning: Clean ~20.0%–21.1%, 8 kHz ~23.2%–29.5%, 10 dB ~32.6%–34.7%, and 5 dB ~42.1%–43.7%. Differences across retrains represent empirical variance on a small sample rather than systematic algorithmic progress.
> **Definition of Number Survival:** A number 'survives' if and only if the exact target numeric value appears in the hypothesis bounded by word boundaries (`\b`) or is successfully parsed by `parse_hindi_number` to within $\pm 0.1$ tolerance (never by raw unconstrained substring).

#### Headline Metric: Held-Out Generalization (N = 7 Utterances, 28 Audio Clips)
| Condition | Held-Out Raw WER (%) | Held-Out Denoised WER (%) | Held-Out Raw Number Acc | Held-Out Denoised Number Acc |
| :--- | :---: | :---: | :---: | :---: |
| clean | 21.05 | 22.11 | 10/10 (100.0%) | 10/10 (100.0%) |
| telephony_8khz | 23.16 | 23.16 | 10/10 (100.0%) | 10/10 (100.0%) |
| telephony_10db | 33.68 | 41.05 | 10/10 (100.0%) | 9/10 (90.0%) |
| telephony_5db | 42.11 | 53.68 | 8/10 (80.0%) | 8/10 (80.0%) |

#### Held-Out End-to-End Pipeline Metric (Audio -> Whisper-LoRA -> Extractor -> Slots)
The table below reports end-to-end slot accuracy evaluated by `benchmark/evaluate_held_out_e2e.py` on actual Whisper-LoRA speech transcripts across all 28 held-out clips:

| Condition | Held-Out Clips (N) | Total Evaluated Slots | End-to-End Slot Accuracy | Incident Type Acc | Temperature Slot Acc | Distance Slot Acc | Location Slot Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| clean | 7 | 28 | 23/28 (82.1%) | 6/7 | 7/7 | 6/7 | 4/7 |
| telephony_8khz | 7 | 28 | 22/28 (78.6%) | 6/7 | 7/7 | 7/7 | 2/7 |
| telephony_10db | 7 | 28 | 21/28 (75.0%) | 5/7 | 7/7 | 6/7 | 3/7 |
| telephony_5db | 7 | 28 | 17/28 (60.7%) | 5/7 | 6/7 | 6/7 | 0/7 |

#### Held-Out Utterance Itemized Breakdown & 5 dB Failure Modes
The table below lists all 7 held-out clips, their ground truth numeric slots, and their survival across all channels:

| Held-Out Clip | Target Temp (°C) | Target Dist (km) | Landmark | Clean | 8 kHz | 10 dB | 5 dB SNR | Failure Mode at 5 dB SNR |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| `utt_03` | −14°C | 10 km | Murthal Dhaba | 2/2 | 2/2 | 2/2 | 2/2 | None (survived: `minus 14`, `10 km`) |
| `utt_07` | +15°C | — | Shahbad | 1/1 | 1/1 | 1/1 | 1/1 | None (survived: `15 degree`) |
| `utt_11` | — | 30 km | Vadodara Bypass | 1/1 | 1/1 | 1/1 | 1/1 | None (survived: `30 km`) |
| `utt_15` | — | 68 km | Mumbai Pune Expressway | 1/1 | 1/1 | 1/1 | **0/1** | **Acoustic confusion:** `68` transcribed as `kem 60` |
| `utt_19` | +19°C | 12 km | Ghaziabad Border | 2/2 | 2/2 | 2/2 | 2/2 | None (survived: `19 degree`, `12 km`) |
| `utt_22` | +25°C | 18 km | Rewari Bypass | 2/2 | 2/2 | 2/2 | 2/2 | None (survived: `25 degree`, `18 km`) |
| `utt_25` | +18°C | — | KMP Expressway | 1/1 | 1/1 | 1/1 | **0/1** | **Dropped digits:** `18 degree` masked by noise $\rightarrow$ `gating degree` |
| **Total** | | | | **10/10 (100%)** | **10/10 (100%)** | **10/10 (100%)** | **8/10 (80.0%)** | **2 failed slots at 5 dB SNR** |

#### In-Domain Training Baseline Reference (Memorized Training Set, N = 18 Utterances, 72 Audio Clips)
| Condition | In-Domain Raw WER (%) | In-Domain Denoised WER (%) | In-Domain Raw Number Acc |
| :--- | :---: | :---: | :---: |
| clean | 0.0 | 0.0 | 23/23 (100.0%) |
| telephony_8khz | 0.0 | 0.0 | 23/23 (100.0%) |
| telephony_10db | 0.0 | 0.4 | 23/23 (100.0%) |
| telephony_5db | 0.0 | 10.67 | 23/23 (100.0%) |

*Note: In-domain 23/23 (100%) number accuracy represents fitting/memorization on the 18 training sentences and is included strictly as an overfitted reference baseline.*

#### Key Empirical Diagnostics & Insights:
1. **Critical Number Survival Rate & Noise Alignment:**
   In dispatch workflows, word error on filler conversational phrases (`"bhaiya"`, `"chal raha hai"`) does not cause misdispatch. What is critical is whether numeric slots (temperatures and distances) survive. On clean, 8 kHz, and 10 dB SNR telephony, **10/10 (100%) of critical numbers survived transcription**, dropping to **8/10 (80.0%)** under harsh 5 dB SNR rumble.
2. **Acoustic Defense for Distance Confusions (Digit-by-Digit Spelling):**
   Because telematics targets do not constrain highway distances (unlike temperatures), the system's primary defense against distance confusion (such as 68 $\rightarrow$ 60 km) is **read-back digit-by-digit spelling** (`generate_readback_prompt` outputs e.g. `Mumbai Pune Expressway se 68 (6, 8) km aage`), forcing phonetic disambiguation on confirmation.
3. **Language Prompt Alignment (`language="en"`):**
   Forcing `language="en", task="transcribe"` in both Whisper-LoRA training (`core/train_asr.py`) and inference prevents the decoder from drifting between language partitions under high acoustic noise.
4. **Denoiser Ablation:**
   A spectral denoiser trained on limited local audio data introduces phase distortion and spectral smearing at low SNRs (at 5 dB SNR held-out WER rises from 42.11% raw to 53.68% denoised). Consequently, **direct raw audio into Whisper is the recommended default pipeline**.
5. **Extractor Regression Tests:**
   - **First Author-Written Set (Smoke Test):** The 10 fresh sentences tested in `test_unseen_fresh_sentences_slot_accuracy` were author-written as an internal smoke test, achieving 28/30 slots (93.3% accuracy, with 2 compound landmark truncations: `Manesar Chauraha` $\rightarrow$ `Manesar` and `Karnal Flyover` $\rightarrow$ `Karnal`).
   - **Second Author-Written Regression Set:** To verify extractor robustness across diverse phrasing (different word orders, missing prepositions, colloquial slang, dialectal Hinglish), a second set of 20 test cases was evaluated (`test_blind_twenty_sentences_held_out_evaluation`), achieving **60 of 60 slots (100.0%)**.
   - **End-to-End Pipeline Evaluation:** On real Whisper-LoRA transcripts of the 28 held-out clips, end-to-end slot accuracy achieved **82.1% on clean, 78.6% on 8 kHz, 75.0% on 10 dB, and 60.7% on 5 dB SNR**.

---

### Illustrative Driver Attention Sensitivity Sweep (0% to 20% Reflexive Error Rate)

> **Methodological Note:** The table below is an illustrative parametric sensitivity sweep ($E_{\text{silent}} = \text{Ungated Errors} \times \text{Assumed Rate}$), demonstrating the mathematical boundary conditions of read-back confirmation under varying driver fatigue and distractibility, rather than a physical measurement of driver psychology.
> **Physical Safety Boundary Notice:** The telematics safety gate intercepts large anomalies (>25°C deviation) even if the driver reflexively agrees (`"haan"`). However, **if a driver genuinely mishears the read-back and repeats the wrong number** (`"haan, assi degree confirm"`), the gate accepts it as an intended override. Software safety gates protect against unverified reflexive confirmations, not against human confirmation of misheard values.

| Assumed Driver Error Rate | Clean (16 kHz) Dispatch Acc | Telephony 8 kHz Dispatch Acc | 10 dB SNR Dispatch Acc | 5 dB SNR Dispatch Acc |
| :---: | :---: | :---: | :---: | :---: |
| **0% (Attentive Driver)** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **5% (Minor Distraction)** | **99.8%** | **99.8%** | **98.8%** | **99.0%** |
| **10% (Moderate Noise)** | **99.6%** | **99.6%** | **97.6%** | **98.0%** |
| **15% (Heavy Fatigue)** | **99.4%** | **99.4%** | **96.4%** | **97.0%** |
| **20% (Severe Noise & Fatigue)** | **99.2%** | **99.2%** | **95.2%** | **96.0%** |

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
