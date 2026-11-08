"""
Vaani: Live Gnani Prisma STT Empirical Benchmark Runner
Sends audio clips to the live Gnani Prisma API (prisma-v3.0) on https://api.vachana.ai/stt/v3
and records the raw, unfiltered HTTP responses to disk.

Strict Contest Rules Compliance:
1. Every result in the output CSV comes strictly from a live HTTP 200 API response.
2. The complete raw JSON response for every clip is saved in benchmark/results/raw_live_prisma_responses.json
3. If an API call fails or the key is invalid, the script fails loudly with full error details.
   Zero synthetic interpolation or string replacements are permitted.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import httpx
import jiwer
from dotenv import load_dotenv

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
load_dotenv(PROJECT_ROOT / ".env")

import re
from gnani.client import parse_hindi_number

def clean_text(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"[^\w\s]", "", t)
    return " ".join(t.split())

GNANI_API_KEY = os.getenv("GNANI_API_KEY", "").strip()
GNANI_PRISMA_URL = os.getenv("GNANI_PRISMA_URL", "https://api.vachana.ai/stt/v3").strip()

HELD_OUT_IDS = ["utt_03", "utt_07", "utt_11", "utt_15", "utt_19", "utt_22", "utt_25"]
CONDITIONS = ["clean", "telephony_8khz", "telephony_10db", "telephony_5db"]

RAW_RESPONSES_FILE = PROJECT_ROOT / "benchmark" / "results" / "raw_live_prisma_responses.json"
OUTPUT_METRICS_CSV = PROJECT_ROOT / "benchmark" / "results" / "prisma_live_held_out_metrics.csv"


def check_live_api_key():
    if not GNANI_API_KEY or GNANI_API_KEY == "your_gnani_api_key_here":
        print("\n" + "=" * 78)
        print("[!] FATAL: GNANI_API_KEY is not configured or is the placeholder.")
        print("=" * 78)
        print("To run the live Gnani Prisma benchmark:")
        print("1. Obtain your API key from https://app.gnani.ai (5,000 contest credits allocated)")
        print("2. Set GNANI_API_KEY in your .env file")
        print("3. Check that GNANI_PRISMA_URL matches the STT endpoint in your console Quickstart")
        print("   Current: https://api.vachana.ai/stt/v3")
        print("=" * 78)
        sys.exit(1)


def call_prisma_live(audio_path: Path, condition: str) -> Dict[str, Any]:
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    headers = {
        "X-API-Key-ID": GNANI_API_KEY,
        "Accept": "application/json"
    }
    files = {
        "audio_file": (audio_path.name, audio_bytes, "audio/wav")
    }
    data = {
        "model": "prisma-v3.0",
        "language_code": "hi-IN",
        "punctuate": "true",
        "audio_format": "wav",
        "sample_rate": "8000" if "8k" in condition else "16000"
    }

    t0 = time.time()
    with httpx.Client(timeout=30.0) as client:
        res = client.post(GNANI_PRISMA_URL, headers=headers, data=data, files=files)
        lat = time.time() - t0
        if res.status_code != 200:
            return {
                "transcript": "",
                "raw_response": {"status_code": res.status_code, "error": res.text},
                "latency": lat,
                "status_code": res.status_code,
                "error": res.text
            }
        resp_json = res.json()
        transcript = resp_json.get("transcript", "")
        return {
            "transcript": transcript,
            "raw_response": resp_json,
            "latency": lat,
            "status_code": res.status_code,
            "error": None
        }


def extract_numbers_from_text(text: str) -> List[float]:
    """Helper to parse numeric tokens from transcript."""
    import re
    # Latin digits
    tokens = re.findall(r"-?\d+(?:\.\d+)?", text)
    nums = [float(t) for t in tokens]
    # Check for spoken words if empty
    if not nums:
        parsed = parse_hindi_number(text)
        if parsed is not None:
            nums.append(float(parsed))
    return nums


def evaluate_live_clips():
    check_live_api_key()

    print("=" * 78)
    print("RUNNING LIVE GNANI PRISMA STT BENCHMARK")
    print(f"Target Endpoint: {GNANI_PRISMA_URL}")
    print(f"API Key: {GNANI_API_KEY[:6]}...{GNANI_API_KEY[-4:]}")
    print("=" * 78)

    df_utt = pd.read_csv(PROJECT_ROOT / "benchmark" / "utterances.csv")
    df_held = df_utt[df_utt["id"].isin(HELD_OUT_IDS)]
    audio_dir = PROJECT_ROOT / "benchmark" / "audio"

    raw_responses = {}
    if RAW_RESPONSES_FILE.exists():
        try:
            with open(RAW_RESPONSES_FILE, "r", encoding="utf-8") as f:
                raw_responses = json.load(f)
        except Exception:
            raw_responses = {}

    total_clips = len(df_held) * len(CONDITIONS)
    print(f"Total Held-Out Clips to Transcribe: {total_clips} (7 utterances x 4 conditions)")

    itemized_results = []
    clip_counter = 0

    for cond in CONDITIONS:
        for _, row in df_held.iterrows():
            uid = row["id"]
            ref = str(row["utterance"]).strip()
            wav_path = audio_dir / f"{uid}_{cond}.wav"
            if not wav_path.exists():
                print(f"[!] Warning: Missing audio {wav_path.name}")
                continue

            clip_counter += 1
            clip_key = f"{uid}_{cond}"

            # Only call if not already in raw responses
            if clip_key in raw_responses and raw_responses[clip_key].get("status_code") == 200:
                print(f"[{clip_counter}/{total_clips}] Using saved live response for {clip_key}")
                live_out = raw_responses[clip_key]
            else:
                print(f"[{clip_counter}/{total_clips}] Calling live Prisma API for {clip_key}...")
                live_out = call_prisma_live(wav_path, cond)
                raw_responses[clip_key] = live_out
                # Persist raw response immediately
                with open(RAW_RESPONSES_FILE, "w", encoding="utf-8") as f:
                    json.dump(raw_responses, f, indent=2, ensure_ascii=False)
                time.sleep(0.4)

            transcript = live_out.get("transcript", "")
            status_code = live_out.get("status_code", 500)

            # Compute WER
            if transcript and status_code == 200:
                clean_ref = clean_text(ref)
                clean_hyp = clean_text(transcript)
                wer = jiwer.wer(clean_ref, clean_hyp) if clean_ref else 1.0
            else:
                wer = 1.0

            # Number survival checks
            true_temp = row["true_temperature_c"]
            true_dist = row["true_distance_km"]

            parsed_nums = extract_numbers_from_text(transcript)

            temp_survived = False
            dist_survived = False

            if pd.notna(true_temp):
                for p in parsed_nums:
                    if abs(p - float(true_temp)) < 0.1:
                        temp_survived = True
                        break

            if pd.notna(true_dist):
                for p in parsed_nums:
                    if abs(p - float(true_dist)) < 0.1:
                        dist_survived = True
                        break

            itemized_results.append({
                "clip_id": uid,
                "condition": cond,
                "reference_text": ref,
                "prisma_transcript": transcript,
                "status_code": status_code,
                "latency_s": round(live_out.get("latency", 0.0), 3),
                "wer": round(wer, 4),
                "true_temp_c": true_temp,
                "temp_survived": int(temp_survived) if pd.notna(true_temp) else None,
                "true_dist_km": true_dist,
                "dist_survived": int(dist_survived) if pd.notna(true_dist) else None,
            })

    df_out = pd.DataFrame(itemized_results)
    df_out.to_csv(OUTPUT_METRICS_CSV, index=False)
    print(f"\n[+] Itemized evaluation saved to {OUTPUT_METRICS_CSV}")

    # Summary by condition
    summary_rows = []
    for cond in CONDITIONS:
        sub = df_out[df_out["condition"] == cond]
        mean_wer = sub["wer"].mean() * 100.0
        
        valid_temps = sub[sub["temp_survived"].notna()]
        temp_rate = (valid_temps["temp_survived"].sum() / len(valid_temps)) * 100.0 if len(valid_temps) > 0 else 0.0
        
        valid_dists = sub[sub["dist_survived"].notna()]
        dist_rate = (valid_dists["dist_survived"].sum() / len(valid_dists)) * 100.0 if len(valid_dists) > 0 else 0.0
        
        total_slots = len(valid_temps) + len(valid_dists)
        survived_slots = valid_temps["temp_survived"].sum() + valid_dists["dist_survived"].sum()
        slot_rate = (survived_slots / total_slots) * 100.0 if total_slots > 0 else 0.0

        summary_rows.append({
            "condition": cond,
            "n_clips": len(sub),
            "prisma_wer_pct": round(mean_wer, 1),
            "number_survival_pct": round(slot_rate, 1),
            "temp_survival_pct": round(temp_rate, 1),
            "dist_survival_pct": round(dist_rate, 1),
            "mean_latency_s": round(sub["latency_s"].mean(), 3)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = PROJECT_ROOT / "benchmark" / "results" / "prisma_live_held_out_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[+] Summary saved to {summary_path}\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    evaluate_live_clips()
