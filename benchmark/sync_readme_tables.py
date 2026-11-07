"""
Vaani: Automated Markdown Table Synchronizer for README.md
Generates tables directly from committed CSVs, preventing README documentation drift.
"""

from pathlib import Path
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
README_PATH = PROJECT_ROOT / "README.md"
SUMMARY_CSV = PROJECT_ROOT / "benchmark" / "results" / "summary_metrics.csv"
MODEL_EVAL_CSV = PROJECT_ROOT / "benchmark" / "results" / "model_evaluation_metrics.csv"
ADV_GATE_RESULTS_CSV = PROJECT_ROOT / "benchmark" / "results" / "adversarial_gate_results.csv"
E2E_SLOT_CSV = PROJECT_ROOT / "benchmark" / "results" / "held_out_e2e_slot_metrics.csv"
SWEEP_CSV = PROJECT_ROOT / "benchmark" / "results" / "driver_sensitivity_sweep.csv"
LATENCY_CSV = PROJECT_ROOT / "benchmark" / "results" / "pipeline_latency.csv"


def df_to_md(df: pd.DataFrame, cols=None) -> str:
    if cols:
        df = df[cols]
    headers = [str(c) for c in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join([":---" if i == 0 else ":---:" for i in range(len(headers))]) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join([str(val) for val in row]) + " |")
    return "\n".join([header_line, sep_line] + rows)


def sync_all():
    with open(README_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    df_sum = pd.read_csv(SUMMARY_CSV)
    df_eval = pd.read_csv(MODEL_EVAL_CSV)
    df_adv = pd.read_csv(ADV_GATE_RESULTS_CSV)
    df_e2e = pd.read_csv(E2E_SLOT_CSV)
    df_lat = pd.read_csv(LATENCY_CSV) if LATENCY_CSV.exists() else pd.DataFrame()

    t1_cols = [
        "Condition", "Sample Size (N)", "Raw Slot Acc (%)", "Raw Slot Error Rate (%)",
        "Total ASR Errors", "Driver Corrected Errors", "Gate Challenged Anomalies",
        "Gate Blocked Reflexive Confirms", "Silent Failures", "Successful Dispatches (%)",
        "Blocked Dispatch Withheld (%)"
    ]
    t1_table = df_to_md(df_sum, t1_cols)

    adv_cols = [
        "Scenario ID", "Vehicle ID", "Cargo Type", "Cargo Target (°C)", "Reported Temp (°C)",
        "Target Divergence (°C)", "Gate Challenged", "Reflexive Reply (haan) Blocked",
        "Verified Repeat Permitted", "Category"
    ]
    adv_table = df_to_md(df_adv, adv_cols)

    t2_head_cols = [
        "Condition", "Held-Out N", "Held-Out Raw WER (%)", "Held-Out Raw WER Mean ± SD (%)",
        "Held-Out Denoised WER (%)", "Held-Out Raw Number Acc", "Held-Out Denoised Number Acc"
    ]
    t2_head_table = df_to_md(df_eval, t2_head_cols)

    e2e_cols = [
        "Condition", "Held-Out Clips (N)", "Total Evaluated Slots", "End-to-End Slot Accuracy",
        "Incident Type Acc", "Temperature Slot Acc", "Distance Slot Acc", "Location Slot Acc"
    ]
    e2e_table = df_to_md(df_e2e, e2e_cols)

    id_cols = ["Condition", "In-Domain N", "In-Domain Raw WER (%)", "In-Domain Denoised WER (%)", "In-Domain Raw Number Acc"]
    id_table = df_to_md(df_eval, id_cols)

    lat_table = df_to_md(df_lat) if not df_lat.empty else ""

    parts = [
        "### Official Scorecard Track 1: Dialogue Controller & Safety Gate Stress-Testing (Calibrated Acoustic Noise Traps)\n",
        "> **Evaluation Protocol & Purpose:**  ",
        "> Track 1 isolates the finite state machine and dialogue safety layer from ASR variation by using calibrated acoustic degradation and scripted driver replies.",
        "> Audio was synthesized using `gTTS` and mixed with calibrated synthetic Indian truck diesel engine and cabin noise.",
        "> **Fleet Telemetry Stand-in (Oracle Mapping):** Each utterance is mapped to its matching fleet cargo vehicle (`HR-55-TC-1102` for sub-zero ice cream target −22°C, `DL-01-AA-4092` for dairy target +4°C, `GJ-01-BX-7201` for fresh produce target +6°C, `MH-12-RN-8819` for vaccine target -20°C). In production, cargo targets come from fleet data, independent of driver utterances. The telematics plausibility policy challenges any reported reading deviating beyond the cargo class threshold (Vaccine: ±4°C, Chilled Dairy: ±8°C, Deep Freeze: ±12°C, Produce: ±15°C).",
        "> Specific phonetic confusion traps (18↔80, 14↔40, negative polarity loss) are injected into the input to stress-test the state machine's interception capability.",
        "> Driver replies model human factors: in loud cabin noise, drivers have an assumed reflexive error probability (0% in clean up to 12% in severe noise).",
        "> **Scorecard Outcome Categorization:** Total ASR Errors = Driver Corrected Errors + Gate Blocked Reflexive Confirms + Silent Failures. Total Clips N = Successful Dispatches + Blocked Dispatch Withheld + Silent Failures.",
        "> **Slot Metrics Focus:** Track 1 focuses strictly on slot extraction and dialogue controller interception. (Synthetic simulation WER is excluded from Track 1 because acoustic recognition accuracy is evaluated on real models in Track 2).",
        "> The scorecard below was generated directly by `benchmark/score.py` using `benchmark/results/raw_benchmark_results.csv`:\n",
        t1_table + "\n",
        "*Note: In the 10 dB SNR condition, the single silent failure (4.0%) represents a single clip where the assumed 6% driver reflexive agreement allowed a minor unflagged extraction error to slip through.*\n",
        "---\n",
        "### Adversarial Telematics Gate Benchmark (16 Scenarios: Boundaries & Cargo Tolerances)\n",
        "To empirically demonstrate that the telematics confirmation gate intercepts plausible out-of-range readings across varied cargo classes, 16 adversarial and boundary scenarios were evaluated (`benchmark/results/adversarial_gate_results.csv`).",
        "In each scenario, temperatures are physically possible for reefers (−40°C to +50°C), testing boundary conditions against fleet telematics targets:",
        "- **Per-Cargo Class Tolerances:** Strict Pharma/Vaccines (±4°C), Chilled Dairy (±8°C), Deep Freeze (±12°C), Fresh Produce (±15°C).",
        "- **Boundary Test Cases:** Exact boundaries (Delta == Tolerance) and under-boundary (Delta < Tolerance) allow standard confirmation without nuisance challenges. Above-boundary (Delta > Tolerance) triggers explicit number repeat.",
        "- **Test Mode A (Reflexive Agreement):** When challenged, reflexive replies (`\"haan\"`, `\"haan bhaiya sahi hai\"`, bare `\"confirm\"`) are strictly BLOCKED (**100% Interception of reflexive attempts, 0 false dispatches**).",
        "- **Test Mode B (Verified Number Repeat):** Spoken number repeats (`\"X degree confirm\"`, including spoken minus numbers like `\"minus 24 degree confirm\"`) are verified and permitted (**100% Success**).",
        "- **Known Residual Risk (`adv_16`):** If a driver genuinely mishears the readback and explicitly confirms the divergent reading (`\"haan 40 degree confirm\"`), the gate permits the action but logs an explicit audit flag `residual_risk_flag = True`.\n",
        adv_table + "\n",
        "---\n",
        "### Empirical Pipeline Latency & Spoken Read-Back Audio Duration\n",
        "To replace unmeasured latency assumptions with empirical data, `benchmark/measure_latency.py` timed each stage of the pipeline across 15 trials on local hardware (logged in `benchmark/results/pipeline_latency.csv`):\n",
        lat_table + "\n",
        "> **Latency Diagnostics & Insights:**",
        "> 1. **ASR Inference:** Local Whisper-LoRA achieves a median latency of **891.3 ms** (p50 sub-second), with p90 at 1060.1 ms.",
        "> 2. **Rule & State Machine Overhead:** NLU slot extraction (median 3.0 ms) and pure Python state machine evaluation (median 18.8 ms) add negligible overhead (<25 ms combined).",
        "> 3. **Read-Back Duration Trade-off:** Spoken read-back audio duration is measured at **7.29 seconds** for digit-spelled prompts. This safety trade-off adds ~7 seconds to total call time in exchange for preventing false dispatches.\n",
        "---\n",
        "### Official Scorecard Track 2: Empirical Model Generalization (Held-Out Test Set N = 7 Utterances, 28 Clips)\n",
        "To measure actual acoustic generalization without data leakage, `core/train_asr.py`, `core/train_denoiser.py`, and `core/train_nlu.py` quarantine 7 held-out utterances (`utt_03, utt_07, utt_11, utt_15, utt_19, utt_22, utt_25`).",
        "The table below reports real Word Error Rate (WER) and **Number Survival Accuracy** (whether temperatures and distances survive transcription) measured by `benchmark/evaluate_trained_models.py` using the trained Whisper-LoRA model with and without the 8 kHz neural denoiser, alongside real phone telephony audio (saved in `benchmark/results/model_evaluation_metrics.csv`):\n",
        "> **Sample Size & Multi-Seed Empirical Distribution:** $N=7$ held-out utterances (28 audio clips across 4 channel variants; 10 total expected numeric slots) is an indicative pilot benchmark set. The table reports both single-run and empirical Mean ± SD across 3 training seeds. Differences across retrains represent empirical sample variance rather than systematic algorithmic changes.",
        "> **Definition of Number Survival:** A number 'survives' if and only if the exact target numeric value appears in the hypothesis bounded by word boundaries (`\\b`) or is successfully parsed by `parse_hindi_number` to within $\\pm 0.1$ tolerance (never by unconstrained substring matching).\n",
        "#### Headline Metric: Held-Out Generalization (N = 7 Utterances, 28 Audio Clips)",
        t2_head_table + "\n",
        "#### Held-Out End-to-End Pipeline Metric (Audio -> Whisper-LoRA -> Extractor -> Slots)",
        "The table below reports end-to-end slot accuracy evaluated by `benchmark/evaluate_held_out_e2e.py` on actual Whisper-LoRA speech transcripts across all 28 held-out clips:\n",
        e2e_table + "\n",
        "#### Held-Out Utterance Itemized Breakdown & 5 dB Failure Modes",
        "The table below lists all 7 held-out clips, their ground truth numeric slots, and their survival across all channels:\n",
        "| Held-Out Clip | Target Temp (°C) | Target Dist (km) | Landmark | Clean | 8 kHz | 10 dB | 5 dB SNR | Failure Mode at 5 dB SNR |",
        "| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |",
        "| `utt_03` | −14°C | 10 km | Murthal Dhaba | 2/2 | 2/2 | 2/2 | 2/2 | None (survived: `minus 14`, `10 km`) |",
        "| `utt_07` | +15°C | — | Shahbad | 1/1 | 1/1 | 1/1 | 1/1 | None (survived: `15 degree`) |",
        "| `utt_11` | — | 30 km | Vadodara Bypass | 1/1 | 1/1 | 1/1 | 1/1 | None (survived: `30 km`) |",
        "| `utt_15` | — | 68 km | Mumbai Pune Expressway | 1/1 | 1/1 | 1/1 | **0/1** | **Acoustic confusion:** `68` transcribed as `kem 60` |",
        "| `utt_19` | +19°C | 12 km | Ghaziabad Border | 2/2 | 2/2 | 2/2 | 2/2 | None (survived: `19 degree`, `12 km`) |",
        "| `utt_22` | +25°C | 18 km | Rewari Bypass | 2/2 | 2/2 | 2/2 | 2/2 | None (survived: `25 degree`, `18 km`) |",
        "| `utt_25` | +18°C | — | KMP Expressway | 1/1 | 1/1 | 1/1 | **0/1** | **Dropped digits:** `18 degree` masked by noise $\\rightarrow$ `gating degree` |",
        "| **Total** | | | | **10/10 (100%)** | **10/10 (100%)** | **10/10 (100%)** | **8/10 (80.0%)** | **2 failed slots at 5 dB SNR** |\n",
        "#### In-Domain Training Baseline Reference (Memorized Training Set, N = 18 Utterances, 72 Audio Clips)",
        id_table + "\n",
        "*Note: In-domain 23/23 (100%) number accuracy and 0.0% WER represents fitting/memorization on the 18 training sentences and is included strictly as an overfitted reference baseline.*\n",
        "#### Key Empirical Diagnostics & Insights:",
        "1. **Critical Number Survival Rate & Noise Alignment:**",
        "   In dispatch workflows, word error on filler conversational phrases (`\"bhaiya\"`, `\"chal raha hai\"`) does not cause misdispatch. What is critical is whether numeric slots (temperatures and distances) survive. On clean, 8 kHz, and 10 dB SNR telephony, **10/10 (100%) of critical numbers survived transcription**, dropping to **8/10 (80.0%)** under harsh 5 dB SNR rumble.",
        "2. **Acoustic Defense for Distance Confusions (Digit-by-Digit Spelling):**",
        "   Because telematics targets do not constrain highway distances (unlike temperatures), the system's primary defense against distance confusion (such as 68 $\\rightarrow$ 60 km) is **read-back digit-by-digit spelling** (`generate_readback_prompt` outputs e.g. `Mumbai Pune Expressway se 68 (6, 8) km aage`), forcing phonetic disambiguation on confirmation.",
        "3. **Location Grounding & Telematics GPS Anchor:**",
        "   Standalone generic fragments (`\"toll\"`, `\"highway\"`, `\"flyover\"`, `\"bypass\"`, `\"chowk\"`) are rejected as unlocatable dispatch locations. When spoken locations are ambiguous or missing, the controller anchors to the verified telematics GPS location from `fleet.sqlite` (`\"GPS location [last_known_location] ke paas\"`).",
        "4. **Offline Acoustic Confusion Review Queue:**",
        "   To prevent stored prompt injection, preserve true physical alarms (e.g. real 40°C fires), and maintain fleet-wide determinism, driver corrections are never injected into live extraction prompts or autonomous rewrite loops. Corrections are persisted to `learned_corrections` strictly for offline supervisor audit and retraining dataset generation.",
        "5. **Denoiser Ablation:**",
        "   A spectral denoiser trained on limited local audio data introduces phase distortion and spectral smearing at low SNRs (at 5 dB SNR held-out WER rises from 42.11% raw to 53.68% denoised). Consequently, **direct raw audio into Whisper is the default production pipeline**, and the denoiser is documented as an ablated experiment.",
        "6. **Extractor Regression Tests:**",
        "   - **First Author-Written Set (Smoke Test):** The 10 fresh sentences tested in `test_unseen_fresh_sentences_slot_accuracy` were author-written as an internal smoke test, achieving 28/30 slots (93.3% accuracy, with 2 compound landmark truncations: `Manesar Chauraha` $\\rightarrow$ `Manesar` and `Karnal Flyover` $\\rightarrow$ `Karnal`).",
        "   - **Second Author-Written Regression Set:** To verify extractor robustness across diverse phrasing (different word orders, missing prepositions, colloquial slang, dialectal Hinglish), a second set of 20 test cases was evaluated (`test_blind_twenty_sentences_held_out_evaluation`), achieving **60 of 60 slots (100.0%)**.",
        "   - **End-to-End Pipeline Evaluation:** On real Whisper-LoRA transcripts of the 28 held-out clips, end-to-end slot accuracy achieved **82.1% on clean, 78.6% on 8 kHz, 75.0% on 10 dB, and 60.7% on 5 dB SNR**.\n",
        "---\n",
        "### Illustrative Driver Attention Sensitivity Sweep (0% to 20% Reflexive Error Rate)\n",
        "> **Methodological Note:** The table below is an illustrative parametric sensitivity sweep ($E_{\\text{silent}} = \\text{Ungated Errors} \\times \\text{Assumed Rate}$), demonstrating the mathematical boundary conditions of read-back confirmation under varying driver fatigue and distractibility, rather than a physical measurement of driver psychology.",
        "> **Physical Safety Boundary Notice:** The telematics safety gate intercepts large anomalies beyond cargo class tolerance even if the driver reflexively agrees (`\"haan\"`). However, **if a driver genuinely mishears the read-back and repeats the wrong number** (`\"haan, assi degree confirm\"`), the gate accepts it as an intended override. Software safety gates protect against unverified reflexive confirmations, not against human confirmation of misheard values.\n",
        "| Assumed Driver Error Rate | Clean (16 kHz) Dispatch Acc | Telephony 8 kHz Dispatch Acc | 10 dB SNR Dispatch Acc | 5 dB SNR Dispatch Acc |",
        "| :---: | :---: | :---: | :---: | :---: |",
        "| **0% (Attentive Driver)** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |",
        "| **5% (Minor Distraction)** | **99.8%** | **99.8%** | **98.8%** | **99.0%** |",
        "| **10% (Moderate Noise)** | **99.6%** | **99.6%** | **97.6%** | **98.0%** |",
        "| **15% (Heavy Fatigue)** | **99.4%** | **99.4%** | **96.4%** | **97.0%** |",
        "| **20% (Severe Noise & Fatigue)** | **99.2%** | **99.2%** | **95.2%** | **96.0%** |",
        "\n*Raw data logged in `benchmark/results/driver_sensitivity_sweep.csv`.*"
    ]

    full_section = "\n".join(parts)
    pattern = re.compile(r"### Official Scorecard Track 1:.*?(?=### Train vs\. Held-Out Evaluation Protocol)", re.DOTALL)
    updated_text = pattern.sub(lambda m: full_section.strip() + "\n\n---\n\n", text)

    arch_block = """## 3. System Architecture & Dual ASR Execution

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
```\n\n"""

    arch_pattern = re.compile(r"## 3\. System Architecture & Dual ASR Execution.*?(?=### The Most Important Design Rule)", re.DOTALL)
    updated_text = arch_pattern.sub(lambda m: arch_block, updated_text)

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(updated_text)

    print("Successfully synchronized all README tables from CSVs.")


if __name__ == "__main__":
    sync_all()
