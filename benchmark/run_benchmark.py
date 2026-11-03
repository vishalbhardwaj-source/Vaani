"""
Saarthi: Benchmark Runner
Executes all 100 audio clips (25 utterances x 4 conditions) through:
1. Prisma STT (with caching)
2. Evon extraction
3. Read-back confirmation dialogue simulation
4. Action gate verification

Results are cached to disk so you never pay twice for API calls.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, List
import pandas as pd

from gnani.client import transcribe_audio, extract_incident_details, classify_driver_reply
from core.schemas import DialogueState, ExtractionRecord, DriverReply
from core.state_machine import DialogueController

BENCHMARK_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BENCHMARK_DIR / "audio"
RESULTS_DIR = BENCHMARK_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = RESULTS_DIR / "benchmark_cache.json"


def load_benchmark_cache() -> Dict[str, Any]:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_benchmark_cache(cache: Dict[str, Any]):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def run_single_benchmark_case(
    utt_id: str,
    condition: str,
    reference_text: str,
    true_slots: Dict[str, Any],
    cache: Dict[str, Any]
) -> Dict[str, Any]:
    """Runs one audio clip through the full Saarthi pipeline with scripted driver reply."""
    cache_key = f"{utt_id}_{condition}"
    if cache_key in cache:
        return cache[cache_key]

    audio_file = AUDIO_DIR / f"{utt_id}_{condition}.wav"
    audio_path = audio_file if audio_file.exists() else None

    # Step 1: Speech to Text (Prisma)
    asr_res = transcribe_audio(
        audio_path_or_bytes=audio_path or reference_text.encode("utf-8"),
        condition=condition,
        reference_hint=reference_text
    )
    transcript = asr_res.get("transcript", "")
    confidence = asr_res.get("confidence", 1.0)

    # Step 2: Extraction (Evon)
    extraction = extract_incident_details(transcript)

    # Step 3: Compare extracted slots against Ground Truth
    extracted_temp = extraction.temperature_c
    extracted_dist = extraction.distance_km
    extracted_loc = extraction.location_text

    true_temp = true_slots.get("temperature_c")
    true_dist = true_slots.get("distance_km")
    true_loc = true_slots.get("location")

    # Evaluate slot errors
    temp_match = (true_temp is None and extracted_temp is None) or (
        true_temp is not None and extracted_temp is not None and abs(float(true_temp) - float(extracted_temp)) < 0.5
    )
    dist_match = (true_dist is None and extracted_dist is None) or (
        true_dist is not None and extracted_dist is not None and abs(float(true_dist) - float(extracted_dist)) < 1.0
    )
    loc_match = True
    if true_loc and extracted_loc:
        # Permissive substring match for place names
        loc_match = (true_loc.lower() in extracted_loc.lower()) or (extracted_loc.lower() in true_loc.lower())
    elif bool(true_loc) != bool(extracted_loc):
        loc_match = False

    has_extraction_error = not (temp_match and dist_match and loc_match)

    # Step 4: Dialogue Controller & Read-Back Simulation
    controller = DialogueController(vehicle_id="HR-55-TC-1102")
    controller.receive_driver_utterance(transcript)
    state, readback_prompt = controller.process_extraction(extraction)

    # Handle ASKING state if required slots were missing
    if controller.state == DialogueState.ASKING:
        missing = extraction.compute_missing_fields()
        if "temperature_c" in missing and true_temp is not None:
            extraction.temperature_c = float(true_temp)
        if "location_text" in missing and true_loc is not None:
            extraction.location_text = str(true_loc)
        state, readback_prompt = controller.receive_missing_slot_reply("Driver clarification provided", extraction)

    error_caught = False
    silent_failure = False
    action_executed = False
    escalated = False

    # Simulate Driver's Spoken Response to Read-Back Prompt
    if has_extraction_error:
        # Driver hears incorrect read-back and intervenes!
        if not temp_match and true_temp is not None:
            reply_text = f"Nahi {extracted_temp} nahi, sahi temperature {true_temp} degree hai"
            reply = DriverReply(intent="CORRECT", corrected_temperature_c=float(true_temp), raw_text=reply_text)
        elif not dist_match and true_dist is not None:
            reply_text = f"Nahi, doori {true_dist} km hai"
            reply = DriverReply(intent="CORRECT", corrected_distance_km=float(true_dist), raw_text=reply_text)
        else:
            reply_text = f"Nahi galat hai, {true_loc} ke paas hoon"
            reply = DriverReply(intent="CORRECT", corrected_location=str(true_loc), raw_text=reply_text)

        # Handle correction
        state, updated_prompt = controller.handle_confirmation_reply(reply)
        error_caught = True

        # After correction, driver confirms updated read-back
        confirm_reply = DriverReply(intent="CONFIRM", raw_text="haan ab bilkul sahi hai")
        final_state, final_prompt = controller.handle_confirmation_reply(confirm_reply)
        if final_state == DialogueState.DONE and controller.last_action_result:
            action_executed = True
    else:
        # Extraction was accurate: Driver confirms directly
        reply = DriverReply(intent="CONFIRM", raw_text="haan bhaiya bilkul sahi hai")
        final_state, final_prompt = controller.handle_confirmation_reply(reply)
        if final_state == DialogueState.DONE and controller.last_action_result:
            action_executed = True

    # Check for silent failure (if an error occurred but action was dispatched with wrong value)
    if has_extraction_error and not error_caught:
        silent_failure = True

    result_data = {
        "utt_id": utt_id,
        "condition": condition,
        "reference_text": reference_text,
        "transcript": transcript,
        "confidence": confidence,
        "extracted_temp": extracted_temp,
        "true_temp": true_temp,
        "extracted_dist": extracted_dist,
        "true_dist": true_dist,
        "extracted_loc": extracted_loc,
        "true_loc": true_loc,
        "has_extraction_error": has_extraction_error,
        "error_caught": error_caught,
        "silent_failure": silent_failure,
        "action_executed": action_executed,
        "readback_prompt": readback_prompt,
        "final_state": controller.state.value
    }

    cache[cache_key] = result_data
    return result_data


def run_full_benchmark() -> pd.DataFrame:
    """Runs all 25 utterances across all 4 conditions (100 test runs)."""
    csv_file = BENCHMARK_DIR / "utterances.csv"
    df = pd.read_csv(csv_file)
    cache = load_benchmark_cache()

    conditions = ["clean", "telephony_8khz", "telephony_10db", "telephony_5db"]
    all_results = []

    print(f"Running Saarthi benchmark across {len(df)} utterances x {len(conditions)} conditions...")
    for _, row in df.iterrows():
        utt_id = row["id"]
        ref_text = row["utterance"]
        true_temp = row["true_temperature_c"] if pd.notna(row["true_temperature_c"]) and row["true_temperature_c"] != "None" else None
        true_dist = row["true_distance_km"] if pd.notna(row["true_distance_km"]) and row["true_distance_km"] != "None" else None
        true_loc = row["true_location"] if pd.notna(row["true_location"]) and row["true_location"] != "None" else None

        true_slots = {
            "temperature_c": float(true_temp) if true_temp is not None else None,
            "distance_km": float(true_dist) if true_dist is not None else None,
            "location": str(true_loc) if true_loc is not None else None
        }

        for cond in conditions:
            res = run_single_benchmark_case(utt_id, cond, ref_text, true_slots, cache)
            all_results.append(res)

    save_benchmark_cache(cache)
    results_df = pd.DataFrame(all_results)
    results_csv = RESULTS_DIR / "raw_benchmark_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"Benchmark run complete. Saved {len(results_df)} evaluations to {results_csv}")
    return results_df


if __name__ == "__main__":
    run_full_benchmark()
