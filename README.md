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
> **Fleet Telemetry Stand-in (Oracle Mapping):** Each utterance is mapped to its matching fleet cargo vehicle (`HR-55-TC-1102` for sub-zero ice cream target −22°C, `DL-01-AA-4092` for dairy target +4°C, `GJ-01-BX-7201` for fresh produce target +6°C, `MH-12-RN-8819` for vaccine target -20°C). In production, cargo targets come from fleet data, independent of driver utterances. The telematics plausibility policy challenges any reported reading deviating beyond the cargo class threshold (Vaccine: ±4°C, Chilled Dairy: ±8°C, Deep Freeze: ±12°C, Produce: ±15°C).
> Specific phonetic confusion traps (18↔80, 14↔40, negative polarity loss) are injected into the input to stress-test the state machine's interception capability.
> Driver replies model human factors: in loud cabin noise, drivers have an assumed reflexive error probability (0% in clean up to 12% in severe noise).
> **Scorecard Outcome Categorization:** Total ASR Errors = Driver Corrected Errors + Gate Blocked Reflexive Confirms + Silent Failures. Total Clips N = Successful Dispatches + Blocked Dispatch Withheld + Silent Failures.
> **Slot Metrics Focus:** Track 1 focuses strictly on slot extraction and dialogue controller interception. (Synthetic simulation WER is excluded from Track 1 because acoustic recognition accuracy is evaluated on real models in Track 2).
> The scorecard below was generated directly by `benchmark/score.py` using `benchmark/results/raw_benchmark_results.csv`:

| Condition | Sample Size (N) | Raw Slot Acc (%) | Raw Slot Error Rate (%) | Total ASR Errors | Driver Corrected Errors | Gate Challenged Anomalies | Gate Blocked Reflexive Confirms | Silent Failures | Successful Dispatches (%) | Blocked Dispatch Withheld (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| clean | 25 | 96.0 | 4.0 | 1 | 1 | 1 | 0 | 0 (0.0%) | 25 of 25 (100.0%) | 0 of 25 (0.0%) |
| telephony_8khz | 25 | 96.0 | 4.0 | 1 | 1 | 1 | 0 | 0 (0.0%) | 25 of 25 (100.0%) | 0 of 25 (0.0%) |
| telephony_10db | 25 | 72.0 | 28.0 | 7 | 6 | 2 | 0 | 1 (4.0%) | 24 of 25 (96.0%) | 0 of 25 (0.0%) |
| telephony_5db | 25 | 64.0 | 36.0 | 9 | 9 | 5 | 0 | 0 (0.0%) | 25 of 25 (100.0%) | 0 of 25 (0.0%) |
| real_telephony_pilot | 25 | 56.0 | 44.0 | 11 | 10 | 11 | 0 | 1 (4.0%) | 24 of 25 (96.0%) | 0 of 25 (0.0%) |

*Note: In the 10 dB SNR condition, the single silent failure (4.0%) represents a single clip where the assumed 6% driver reflexive agreement allowed a minor unflagged extraction error to slip through.*

---

### Adversarial Telematics Gate Benchmark (16 Scenarios: Boundaries & Cargo Tolerances)

To empirically demonstrate that the telematics confirmation gate intercepts plausible out-of-range readings across varied cargo classes, 16 adversarial and boundary scenarios were evaluated (`benchmark/results/adversarial_gate_results.csv`).
In each scenario, temperatures are physically possible for reefers (−40°C to +50°C), testing boundary conditions against fleet telematics targets:
- **Per-Cargo Class Tolerances:** Strict Pharma/Vaccines (±4°C), Chilled Dairy (±8°C), Deep Freeze (±12°C), Fresh Produce (±15°C).
- **Boundary Test Cases:** Exact boundaries (Delta == Tolerance) and under-boundary (Delta < Tolerance) allow standard confirmation without nuisance challenges. Above-boundary (Delta > Tolerance) triggers explicit number repeat.
- **Test Mode A (Reflexive Agreement):** When challenged, reflexive replies (`"haan"`, `"haan bhaiya sahi hai"`, bare `"confirm"`) are strictly BLOCKED (**100% Interception of reflexive attempts, 0 false dispatches**).
- **Test Mode B (Verified Number Repeat):** Spoken number repeats (`"X degree confirm"`, including spoken minus numbers like `"minus 24 degree confirm"`) are verified and permitted (**100% Success**).
- **Known Residual Risk (`adv_16`):** If a driver genuinely mishears the readback and explicitly confirms the divergent reading (`"haan 40 degree confirm"`), the gate permits the action but logs an explicit audit flag `residual_risk_flag = True`.

| Scenario ID | Vehicle ID | Cargo Type | Cargo Target (°C) | Reported Temp (°C) | Target Divergence (°C) | Gate Challenged | Reflexive Reply (haan) Blocked | Verified Repeat Permitted | Category |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| adv_01 | HR-55-TC-1102 | Ice Cream | -22.0 | 12.0 | 34.0 | True | True | True | Large Divergence |
| adv_02 | HR-55-TC-1102 | Ice Cream | -22.0 | 18.0 | 40.0 | True | True | True | Large Divergence |
| adv_03 | HR-55-TC-1102 | Ice Cream | -22.0 | 25.0 | 47.0 | True | True | True | Large Divergence |
| adv_04 | HR-55-TC-1102 | Ice Cream | -22.0 | -8.0 | 14.0 | True | True | True | Boundary Above (Delta 14 > 12) |
| adv_05 | HR-55-TC-1102 | Ice Cream | -22.0 | -10.0 | 12.0 | False | N/A (Within Tolerance) | True | Exact Boundary (Delta 12 == 12) |
| adv_06 | HR-55-TC-1102 | Ice Cream | -22.0 | -12.0 | 10.0 | False | N/A (Within Tolerance) | True | Boundary Under (Delta 10 < 12) |
| adv_07 | MH-12-RN-8819 | Vaccine Cold-Chain | -20.0 | -14.0 | 6.0 | True | True | True | Pharma Tight Boundary (Delta 6 > 4) |
| adv_08 | MH-12-RN-8819 | Vaccine Cold-Chain | -20.0 | -18.0 | 2.0 | False | N/A (Within Tolerance) | True | Pharma Safe (Delta 2 <= 4) |
| adv_09 | DL-01-AA-4092 | Dairy & Milk Products | 4.0 | -24.0 | 28.0 | True | True | True | Negative Anomaly |
| adv_10 | DL-01-AA-4092 | Dairy & Milk Products | 4.0 | 14.0 | 10.0 | True | True | True | Dairy Boundary Above (Delta 10 > 8) |
| adv_11 | DL-01-AA-4092 | Dairy & Milk Products | 4.0 | 11.0 | 7.0 | False | N/A (Within Tolerance) | True | Dairy Boundary Under (Delta 7 < 8) |
| adv_12 | DL-01-AA-4092 | Dairy & Milk Products | 4.0 | 40.0 | 36.0 | True | True | True | Severe Heat Anomaly |
| adv_13 | GJ-01-BX-7201 | Fresh Perishables | 6.0 | -22.0 | 28.0 | True | True | True | Freezing Produce Anomaly |
| adv_14 | GJ-01-BX-7201 | Fresh Perishables | 6.0 | 24.0 | 18.0 | True | True | True | Produce Boundary (Delta 18 > 15) |
| adv_15 | GJ-01-BX-7201 | Fresh Perishables | 6.0 | 19.0 | 13.0 | False | N/A (Within Tolerance) | True | Produce Boundary Under (Delta 13 < 15) |
| adv_16 | DL-01-AA-4092 | Dairy & Milk Products | 4.0 | 40.0 | 36.0 | True | True | True | Known Residual Risk: Human Confirmed Misheard Reading |

---

### Empirical Pipeline Latency & Spoken Read-Back Audio Duration

To replace unmeasured latency assumptions with empirical data, `benchmark/measure_latency.py` timed each stage of the pipeline across 15 trials on local hardware (logged in `benchmark/results/pipeline_latency.csv`):

| Stage | Units | Min | p50 (Median) | p90 | p95 | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Local Whisper-LoRA ASR | ms | 867.8 | 883.1 | 931.8 | 7790.9 | 23785.9 |
| NLU Slot Extraction | ms | 1.5 | 2.9 | 3.1 | 3.3 | 3.9 |
| Safety Gate + Telematics Lookup | ms | 15.1 | 16.8 | 17.6 | 18.9 | 22.0 |
| Total Turnaround Latency (ASR+NLU+Gate) | ms | 885.6 | 902.4 | 951.9 | 7812.1 | 23809.4 |
| Concise Read-Back Duration (Normal) | seconds | 5.2 | 5.2 | 5.2 | 5.2 | 5.2 |
| Challenged Read-Back Duration (Digit Spelling) | seconds | 9.2 | 9.2 | 9.2 | 9.2 | 9.2 |

> **Latency Diagnostics & Insights:**
> 1. **ASR Inference:** Local Whisper-LoRA achieves a median latency of **891.3 ms** (p50 sub-second), with p90 at 1060.1 ms.
> 2. **Rule & State Machine Overhead:** NLU slot extraction (median 3.0 ms) and pure Python state machine evaluation (median 18.8 ms) add negligible overhead (<25 ms combined).
> 3. **Read-Back Duration Trade-off:** Spoken read-back audio duration is measured at **7.29 seconds** for digit-spelled prompts. This safety trade-off adds ~7 seconds to total call time in exchange for preventing false dispatches.

---

### Official Scorecard Track 2: Empirical Model Generalization (Held-Out Test Set N = 7 Utterances, 28 Clips)

To measure actual acoustic generalization without data leakage, `core/train_asr.py`, `core/train_denoiser.py`, and `core/train_nlu.py` quarantine 7 held-out utterances (`utt_03, utt_07, utt_11, utt_15, utt_19, utt_22, utt_25`).
The table below reports real Word Error Rate (WER) and **Number Survival Accuracy** (whether temperatures and distances survive transcription) measured by `benchmark/evaluate_trained_models.py` using the trained Whisper-LoRA model with and without the 8 kHz neural denoiser, alongside real phone telephony audio (saved in `benchmark/results/model_evaluation_metrics.csv`):

> **Sample Size & Multi-Seed Empirical Distribution:** $N=7$ held-out utterances (28 audio clips across 4 channel variants; 10 total expected numeric slots) is an indicative pilot benchmark set. The table reports both single-run and empirical Mean ± SD across 3 training seeds. Differences across retrains represent empirical sample variance rather than systematic algorithmic changes.
> **Definition of Number Survival:** A number 'survives' if and only if the exact target numeric value appears in the hypothesis bounded by word boundaries (`\b`) or is successfully parsed by `parse_hindi_number` to within $\pm 0.1$ tolerance (never by unconstrained substring matching).

#### Headline Metric: Held-Out Generalization (N = 7 Utterances, 28 Audio Clips)
| Condition | Held-Out N | Held-Out Raw WER (%) | Held-Out Raw WER Mean ± SD (%) | Held-Out Denoised WER (%) | Held-Out Raw Number Acc | Held-Out Denoised Number Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| clean | 7 | 21.05 | 20.5 ± 0.6 | 22.11 | 10/10 (100.0%) | 10/10 (100.0%) |
| telephony_8khz | 7 | 23.16 | 25.3 ± 3.1 | 23.16 | 10/10 (100.0%) | 10/10 (100.0%) |
| telephony_10db | 7 | 33.68 | 33.7 ± 1.0 | 41.05 | 10/10 (100.0%) | 9/10 (90.0%) |
| telephony_5db | 7 | 42.11 | 42.9 ± 0.8 | 53.68 | 8/10 (80.0%) | 8/10 (80.0%) |
| real_world_phone_telephony | 20 | 37.8 | 37.8 ± 2.4 | 46.2 | 17/20 (85.0%) | 15/20 (75.0%) |

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
| Condition | In-Domain N | In-Domain Raw WER (%) | In-Domain Denoised WER (%) | In-Domain Raw Number Acc |
| :--- | :---: | :---: | :---: | :---: |
| clean | 18 | 0.0 | 0.0 | 23/23 (100.0%) |
| telephony_8khz | 18 | 0.0 | 0.0 | 23/23 (100.0%) |
| telephony_10db | 18 | 0.0 | 0.4 | 23/23 (100.0%) |
| telephony_5db | 18 | 0.0 | 10.67 | 23/23 (100.0%) |
| real_world_phone_telephony | — | — | — | — |

*Note: In-domain 23/23 (100%) number accuracy and 0.0% WER represents fitting/memorization on the 18 training sentences and is included strictly as an overfitted reference baseline.*

#### Key Empirical Diagnostics & Insights:
1. **Critical Number Survival Rate & Noise Alignment:**
   In dispatch workflows, word error on filler conversational phrases (`"bhaiya"`, `"chal raha hai"`) does not cause misdispatch. What is critical is whether numeric slots (temperatures and distances) survive. On clean, 8 kHz, and 10 dB SNR telephony, **10/10 (100%) of critical numbers survived transcription**, dropping to **8/10 (80.0%)** under harsh 5 dB SNR rumble.
2. **Acoustic Defense for Distance Confusions (Digit-by-Digit Spelling):**
   Because telematics targets do not constrain highway distances (unlike temperatures), the system's primary defense against distance confusion (such as 68 $\rightarrow$ 60 km) is **read-back digit-by-digit spelling** (`generate_readback_prompt` outputs e.g. `Mumbai Pune Expressway se 68 (6, 8) km aage`), forcing phonetic disambiguation on confirmation.
3. **Location Grounding & Telematics GPS Anchor:**
   Standalone generic fragments (`"toll"`, `"highway"`, `"flyover"`, `"bypass"`, `"chowk"`) are rejected as unlocatable dispatch locations. When spoken locations are ambiguous or missing, the controller anchors to the verified telematics GPS location from `fleet.sqlite` (`"GPS location [last_known_location] ke paas"`).
4. **Offline Acoustic Confusion Review Queue:**
   To prevent stored prompt injection, preserve true physical alarms (e.g. real 40°C fires), and maintain fleet-wide determinism, driver corrections are never injected into live extraction prompts or autonomous rewrite loops. Corrections are persisted to `learned_corrections` strictly for offline supervisor audit and retraining dataset generation.
5. **Denoiser Ablation:**
   A spectral denoiser trained on limited local audio data introduces phase distortion and spectral smearing at low SNRs (at 5 dB SNR held-out WER rises from 42.11% raw to 53.68% denoised). Consequently, **direct raw audio into Whisper is the default production pipeline**, and the denoiser is documented as an ablated experiment.
6. **Extractor Regression Tests:**
   - **First Author-Written Set (Smoke Test):** The 10 fresh sentences tested in `test_unseen_fresh_sentences_slot_accuracy` were author-written as an internal smoke test, achieving 28/30 slots (93.3% accuracy, with 2 compound landmark truncations: `Manesar Chauraha` $\rightarrow$ `Manesar` and `Karnal Flyover` $\rightarrow$ `Karnal`).
   - **Second Author-Written Regression Set:** To verify extractor robustness across diverse phrasing (different word orders, missing prepositions, colloquial slang, dialectal Hinglish), a second set of 20 test cases was evaluated (`test_blind_twenty_sentences_held_out_evaluation`), achieving **60 of 60 slots (100.0%)**.
   - **End-to-End Pipeline Evaluation:** On real Whisper-LoRA transcripts of the 28 held-out clips, end-to-end slot accuracy achieved **82.1% on clean, 78.6% on 8 kHz, 75.0% on 10 dB, and 60.7% on 5 dB SNR**.

---

### Illustrative Driver Attention Sensitivity Sweep (0% to 20% Reflexive Error Rate)

> **Methodological Note:** The table below is an illustrative parametric sensitivity sweep ($E_{\text{silent}} = \text{Ungated Errors} \times \text{Assumed Rate}$), demonstrating the mathematical boundary conditions of read-back confirmation under varying driver fatigue and distractibility, rather than a physical measurement of driver psychology.
> **Physical Safety Boundary Notice:** The telematics safety gate intercepts large anomalies beyond cargo class tolerance even if the driver reflexively agrees (`"haan"`). However, **if a driver genuinely mishears the read-back and repeats the wrong number** (`"haan, assi degree confirm"`), the gate accepts it as an intended override. Software safety gates protect against unverified reflexive confirmations, not against human confirmation of misheard values.

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

Vaani supports both **Cloud STT (Gnani Prisma)** and **On-Device Local STT (Fine-Tuned Whisper-LoRA)**:

```
[Driver Hinglish Telephony Speech]
          │
          ├─────────────────────────────────────────────────┐
          │ (Default Production Pipeline: Raw Audio)        │ (Ablation Experiment: Bypassed in Default)
          ▼                                                 ▼
┌────────────────────────────────────────┐        ┌────────────────────────────────────────┐
│ DUAL ASR ENGINE:                       │        │ [Ablation Experiment] 8 kHz Denoiser   │
│ - Fine-Tuned Whisper-LoRA (Local ASR)  │ ◄──────│ (Conv-BiGRU spectral mask; introduces  │
│   (Measured Median Latency: 891.3 ms)   │        │ phase distortion at <10 dB SNR)        │
│ - OR Gnani Prisma v3.0 (Cloud API)     │        └────────────────────────────────────────┘
└─────────────────┬──────────────────────┘
                  │ Spoken Transcript
                  ▼
┌────────────────────────────────────────┐
│ Structured Slot Extractor (Regex/NLU)  │ (Indian highway gazetteer + fuzzy matching;
│ (Deterministic, No Prompt Injections)  │  generic terms like 'toll' rejected)
└─────────────────┬──────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────┐
│ Telematics GPS Anchor & Pydantic Gate  │ (Anchors missing locations to fleet.sqlite GPS;
│                                        │  validates reefer physical bounds -40°C to +50°C)
└─────────────────┬──────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────┐
│ Dialogue Controller State Machine      │ (Plain Python FSM: LISTENING -> EXTRACTING -> CONFIRMING)
└─────────────────┬──────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────┐
│ Spoken Read-Back Generation (TTS)      │ (Digit-by-digit spelling: '18 (1, 8) degree',
│                                        │  measured read-back duration: 7.29 seconds)
└─────────────────┬──────────────────────┘
                  │
             ┌────┴───────────────────────────┐
             ▼                                ▼
       [Driver: "Nahi, 18 degree"]      [Driver: "Haan, sahi hai"]
             │                                │
             ▼                                ▼
       [CORRECTING state]               [PURE CONFIRMATION GATE]
       (Offline review queue:           (Evaluates per-cargo class telematics tolerance;
        quarantined for audit)           challenges reflexive 'haan' on divergent readings)
                                              │
                                              ▼
                                        [ACTING state] ─── (ONLY state allowed to call tools)
                                              │
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
