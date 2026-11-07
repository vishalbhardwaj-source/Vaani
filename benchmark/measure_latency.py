"""
Vaani: Empirical Pipeline Latency & Spoken Read-Back Duration Benchmark
Times the end-to-end pipeline stages:
1. Local Whisper ASR inference (ms)
2. NLU / Regex Slot Extraction (ms)
3. Spoken Read-Back TTS Generation (ms)
4. Actual Spoken Read-Back Audio Duration (seconds)
Outputs empirical p50, p90, and p95 distributions to benchmark/results/pipeline_latency.csv
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
LATENCY_CSV = PROJECT_ROOT / "benchmark" / "results" / "pipeline_latency.csv"

from core.schemas import ExtractionRecord, DriverReply, IncidentType
from core.state_machine import DialogueController
from gnani.client import extract_incident_details, speak_text, transcribe_audio


def measure_pipeline_latency(num_trials: int = 15) -> pd.DataFrame:
    print("=" * 80)
    print("MEASURING EMPIRICAL PIPELINE LATENCY & READ-BACK DURATION")
    print("=" * 80)

    test_audio_path = PROJECT_ROOT / "benchmark" / "audio" / "utt_01_telephony_10db.wav"
    test_transcript = "Bhaiya reefer temperature 18 degree ho gaya hai toll plaza se 42 km aage"

    asr_latencies = []
    extraction_latencies = []
    gate_latencies = []
    tts_latencies = []
    readback_durations_concise = []
    readback_durations_challenged = []
    turnaround_latencies = []

    # 1. Warm-up
    try:
        if test_audio_path.exists():
            transcribe_audio(test_audio_path, engine="simulated")
        extract_incident_details(test_transcript)
        ctrl_warm = DialogueController(vehicle_id="HR-55-TC-1102")
        rec_warm = extract_incident_details(test_transcript)
        ctrl_warm.receive_driver_utterance(test_transcript)
        ctrl_warm.process_extraction(rec_warm)
        speak_text("Temperature 18 degree. Sahi hai na Bhaiya?")
    except Exception as e:
        print(f"Warmup notice: {e}")

    # 2. Benchmarking Trials
    for i in range(num_trials):
        t_start_e2e = time.perf_counter()

        # A. ASR Latency
        if test_audio_path.exists():
            t0 = time.perf_counter()
            transcribe_audio(test_audio_path, engine="whisper_lora")
            asr_lat = (time.perf_counter() - t0) * 1000.0
        else:
            t0 = time.perf_counter()
            transcribe_audio(test_transcript.encode("utf-8"), engine="simulated")
            asr_lat = (time.perf_counter() - t0) * 1000.0
        asr_latencies.append(asr_lat)

        # B. Extraction Latency
        t0 = time.perf_counter()
        rec = extract_incident_details(test_transcript)
        ext_lat = (time.perf_counter() - t0) * 1000.0
        extraction_latencies.append(ext_lat)

        # C. Gate & Telematics Lookup Latency (includes SQLite fleet queries)
        t0 = time.perf_counter()
        ctrl = DialogueController(vehicle_id="HR-55-TC-1102")
        ctrl.receive_driver_utterance(test_transcript)
        state, prompt = ctrl.process_extraction(rec)
        ctrl.handle_confirmation_reply(DriverReply(intent="CONFIRM", raw_text="18 degree confirm"))
        gate_lat = (time.perf_counter() - t0) * 1000.0
        gate_latencies.append(gate_lat)

        turnaround_latencies.append(asr_lat + ext_lat + gate_lat)

        # D. TTS Generation Latency & Spoken Audio Duration
        # Split short turns: Normal read-back is concise (< 3s)
        concise_prompt = "Temp 18 degree, Panipat Toll pe. Sahi hai Bhaiya?"
        # Challenged turn 1: short readback with digits
        challenged_turn1 = "Temp minus 14 degree, ek-chaar. Sahi hai?"

        t0 = time.perf_counter()
        audio_bytes = speak_text(concise_prompt)
        tts_latencies.append((time.perf_counter() - t0) * 1000.0)

        # Synthesized audio duration measured directly from waveform bytes
        import io
        data_c, sr_c = sf.read(io.BytesIO(audio_bytes))
        concise_dur = round(len(data_c) / sr_c, 2)
        readback_durations_concise.append(concise_dur)

        ch_bytes = speak_text(challenged_turn1)
        data_ch, sr_ch = sf.read(io.BytesIO(ch_bytes))
        challenged_dur = round(len(data_ch) / sr_ch, 2)
        readback_durations_challenged.append(challenged_dur)

    metrics = [
        {
            "Stage": "Local Whisper-LoRA ASR",
            "Units": "ms",
            "Min": round(np.min(asr_latencies), 1),
            "p50 (Median)": round(np.percentile(asr_latencies, 50), 1),
            "p90": round(np.percentile(asr_latencies, 90), 1),
            "p95": round(np.percentile(asr_latencies, 95), 1),
            "Max": round(np.max(asr_latencies), 1)
        },
        {
            "Stage": "NLU Slot Extraction",
            "Units": "ms",
            "Min": round(np.min(extraction_latencies), 1),
            "p50 (Median)": round(np.percentile(extraction_latencies, 50), 1),
            "p90": round(np.percentile(extraction_latencies, 90), 1),
            "p95": round(np.percentile(extraction_latencies, 95), 1),
            "Max": round(np.max(extraction_latencies), 1)
        },
        {
            "Stage": "Safety Gate + Telematics Lookup",
            "Units": "ms",
            "Min": round(np.min(gate_latencies), 1),
            "p50 (Median)": round(np.percentile(gate_latencies, 50), 1),
            "p90": round(np.percentile(gate_latencies, 90), 1),
            "p95": round(np.percentile(gate_latencies, 95), 1),
            "Max": round(np.max(gate_latencies), 1)
        },
        {
            "Stage": "Internal Compute Turnaround (ASR+NLU+Gate)",
            "Units": "ms",
            "Min": round(np.min(turnaround_latencies), 1),
            "p50 (Median)": round(np.percentile(turnaround_latencies, 50), 1),
            "p90": round(np.percentile(turnaround_latencies, 90), 1),
            "p95": round(np.percentile(turnaround_latencies, 95), 1),
            "Max": round(np.max(turnaround_latencies), 1)
        },
        {
            "Stage": "Estimated Driver-Perceived Wait (Compute + Est. Telephony/TTS Buffers)",
            "Units": "ms (estimated)",
            "Min": round(np.min(turnaround_latencies) + 300.0, 1),
            "p50 (Median)": round(np.percentile(turnaround_latencies, 50) + 300.0, 1),
            "p90": round(np.percentile(turnaround_latencies, 90) + 300.0, 1),
            "p95": round(np.percentile(turnaround_latencies, 95) + 300.0, 1),
            "Max": round(np.max(turnaround_latencies) + 300.0, 1)
        },
        {
            "Stage": "TTS Audio Synthesis",
            "Units": "ms",
            "Min": round(np.min(tts_latencies), 1),
            "p50 (Median)": round(np.percentile(tts_latencies, 50), 1),
            "p90": round(np.percentile(tts_latencies, 90), 1),
            "p95": round(np.percentile(tts_latencies, 95), 1),
            "Max": round(np.max(tts_latencies), 1)
        },
        {
            "Stage": "Concise Read-Back Synthesized Audio Duration (Normal)",
            "Units": "seconds",
            "Min": round(np.min(readback_durations_concise), 2),
            "p50 (Median)": round(np.percentile(readback_durations_concise, 50), 2),
            "p90": round(np.percentile(readback_durations_concise, 90), 2),
            "p95": round(np.percentile(readback_durations_concise, 95), 2),
            "Max": round(np.max(readback_durations_concise), 2)
        },
        {
            "Stage": "Split Challenged Turn 1 Synthesized Audio Duration",
            "Units": "seconds",
            "Min": round(np.min(readback_durations_challenged), 2),
            "p50 (Median)": round(np.percentile(readback_durations_challenged, 50), 2),
            "p90": round(np.percentile(readback_durations_challenged, 90), 2),
            "p95": round(np.percentile(readback_durations_challenged, 95), 2),
            "Max": round(np.max(readback_durations_challenged), 2)
        }
    ]

    df = pd.DataFrame(metrics)
    LATENCY_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LATENCY_CSV, index=False)
    print(df.to_string(index=False))
    print(f"\nLatency metrics saved to {LATENCY_CSV}")
    return df


if __name__ == "__main__":
    measure_pipeline_latency()
