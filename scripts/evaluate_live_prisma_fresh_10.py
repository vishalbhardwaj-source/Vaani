"""
Evaluates live Gnani Prisma STT and Vaani's downstream deterministic extractor on
10 completely fresh, unseen freight dispatch sentences across diverse Indian logistics corridors.
Zero code changes or gazetteer additions are made for this test.
"""

import os
import sys
import json
import time
from pathlib import Path
import pandas as pd
import httpx
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from gnani.client import extract_incident_details, GNANI_API_KEY, GNANI_PRISMA_URL

AUDIO_DIR = PROJECT_ROOT / "benchmark" / "audio" / "fresh_unseen_10"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

RAW_RESPONSES_FILE = PROJECT_ROOT / "benchmark" / "results" / "raw_live_prisma_fresh_10.json"
OUTPUT_METRICS_CSV = PROJECT_ROOT / "benchmark" / "results" / "prisma_fresh_unseen_metrics.csv"
OUTPUT_SUMMARY_CSV = PROJECT_ROOT / "benchmark" / "results" / "prisma_fresh_unseen_summary.csv"

FRESH_SCENARIOS = [
    {
        "id": "fresh_01",
        "sentence": "Sir reefer temperature minus 18 degree ho gaya hai Agra highway par 25 km pehle",
        "true_incident": "reefer_temperature",
        "true_temp": -18.0,
        "true_dist": 25.0,
        "true_loc": "Agra"
    },
    {
        "id": "fresh_02",
        "sentence": "Emergency breakdown fan belt toot gaya hai Mathura toll se 15 km aage",
        "true_incident": "breakdown",
        "true_temp": None,
        "true_dist": 15.0,
        "true_loc": "Mathura"
    },
    {
        "id": "fresh_03",
        "sentence": "Temperature 4 degree chal raha hai Karnal bypass se 8 km pe",
        "true_incident": "reefer_temperature",
        "true_temp": 4.0,
        "true_dist": 8.0,
        "true_loc": "Karnal"
    },
    {
        "id": "fresh_04",
        "sentence": "Gaadi puncture ho gayi hai Sonipat ke paas 22 km door",
        "true_incident": "breakdown",
        "true_temp": None,
        "true_dist": 22.0,
        "true_loc": "Sonipat"
    },
    {
        "id": "fresh_05",
        "sentence": "Chiller temperature minus 5 degree display ho raha hai Jaipur highway km 45",
        "true_incident": "reefer_temperature",
        "true_temp": -5.0,
        "true_dist": 45.0,
        "true_loc": "Jaipur"
    },
    {
        "id": "fresh_06",
        "sentence": "Bhaiya temperature 28 degree pahunch gaya hai Surat bypass se 50 km aage",
        "true_incident": "reefer_temperature",
        "true_temp": 28.0,
        "true_dist": 50.0,
        "true_loc": "Surat"
    },
    {
        "id": "fresh_07",
        "sentence": "Traffic jam hai bhari delay hoga Indore bypass par 12 km pehle",
        "true_incident": "delay",
        "true_temp": None,
        "true_dist": 12.0,
        "true_loc": "Indore"
    },
    {
        "id": "fresh_08",
        "sentence": "Reefer off hai aur temperature minus 22 degree se badh raha hai Ambala cantt 18 km",
        "true_incident": "reefer_temperature",
        "true_temp": -22.0,
        "true_dist": 18.0,
        "true_loc": "Ambala Cantt"
    },
    {
        "id": "fresh_09",
        "sentence": "Breakdown alert engine garam ho raha hai Neemrana industrial area se 30 km door",
        "true_incident": "breakdown",
        "true_temp": None,
        "true_dist": 30.0,
        "true_loc": "Neemrana"
    },
    {
        "id": "fresh_10",
        "sentence": "Temperature 14 degree dikha raha hai Bawal toll plaza se 10 km pehle",
        "true_incident": "reefer_temperature",
        "true_temp": 14.0,
        "true_dist": 10.0,
        "true_loc": "Bawal"
    }
]


def generate_audio_if_needed():
    """Generates audio for each scenario using gTTS."""
    from gtts import gTTS
    for sc in FRESH_SCENARIOS:
        wav_path = AUDIO_DIR / f"{sc['id']}.wav"
        mp3_path = AUDIO_DIR / f"{sc['id']}.mp3"
        if not wav_path.exists():
            print(f"Synthesizing audio for {sc['id']}...")
            tts = gTTS(text=sc["sentence"], lang="hi", slow=False)
            tts.save(str(mp3_path))
            # Convert to standard 16kHz WAV using soundfile / scipy or pydub
            try:
                import soundfile as sf
                from pydub import AudioSegment
                seg = AudioSegment.from_mp3(str(mp3_path)).set_frame_rate(16000).set_channels(1)
                seg.export(str(wav_path), format="wav")
            except Exception:
                # Fallback: direct rename or soundfile read
                import shutil
                shutil.copy(mp3_path, wav_path)


def call_prisma_live(audio_path: Path) -> dict:
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    headers = {
        "X-API-Key-ID": GNANI_API_KEY,
        "Accept": "application/json"
    }
    files = {"audio_file": (audio_path.name, audio_bytes, "audio/wav")}
    data = {
        "model": "prisma-v3.0",
        "language_code": "hi-IN",
        "punctuate": "true",
        "audio_format": "wav",
        "sample_rate": "16000"
    }
    t0 = time.time()
    with httpx.Client(timeout=30.0) as client:
        res = client.post(GNANI_PRISMA_URL, headers=headers, data=data, files=files)
        lat = time.time() - t0
        if res.status_code == 200:
            resp_json = res.json()
            return {
                "status_code": 200,
                "transcript": resp_json.get("transcript", ""),
                "latency_s": round(lat, 3),
                "raw": resp_json,
                "error": None
            }
        else:
            return {
                "status_code": res.status_code,
                "transcript": "",
                "latency_s": round(lat, 3),
                "raw": res.text,
                "error": res.text
            }


def main():
    if not GNANI_API_KEY or GNANI_API_KEY == "your_gnani_api_key_here":
        print("GNANI_API_KEY not configured. Exiting.")
        return

    generate_audio_if_needed()

    raw_responses = {}
    if RAW_RESPONSES_FILE.exists():
        try:
            with open(RAW_RESPONSES_FILE, "r", encoding="utf-8") as f:
                raw_responses = json.load(f)
        except Exception:
            raw_responses = {}

    results = []

    print("=" * 78)
    print("RUNNING LIVE GNANI PRISMA STT ON 10 FRESH UNSEEN SCENARIOS")
    print(f"Target: {GNANI_PRISMA_URL}")
    print("=" * 78)

    for sc in FRESH_SCENARIOS:
        sid = sc["id"]
        wav_path = AUDIO_DIR / f"{sid}.wav"

        if sid in raw_responses and raw_responses[sid].get("status_code") == 200:
            print(f"[{sid}] Using saved live response")
            live_out = raw_responses[sid]
        else:
            print(f"[{sid}] Calling live Prisma API...")
            live_out = call_prisma_live(wav_path)
            raw_responses[sid] = live_out
            with open(RAW_RESPONSES_FILE, "w", encoding="utf-8") as f:
                json.dump(raw_responses, f, indent=2, ensure_ascii=False)
            time.sleep(0.4)

        transcript = live_out.get("transcript", "")
        lat = live_out.get("latency_s", 0.0)

        # Run deterministic extractor on Prisma output
        rec = extract_incident_details(transcript)

        # Score extraction
        true_inc = sc["true_incident"]
        pred_inc = rec.incident_type.value
        inc_ok = (true_inc == pred_inc)

        true_temp = sc["true_temp"]
        pred_temp = rec.temperature_c
        if true_temp is None:
            temp_ok = (pred_temp is None)
        else:
            temp_ok = (pred_temp is not None and abs(float(pred_temp) - float(true_temp)) < 0.1)

        true_dist = sc["true_dist"]
        pred_dist = rec.distance_km
        if true_dist is None:
            dist_ok = (pred_dist is None)
        else:
            dist_ok = (pred_dist is not None and abs(float(pred_dist) - float(true_dist)) < 0.1)

        true_loc = sc["true_loc"]
        pred_loc = rec.location_text
        if not true_loc:
            loc_ok = (pred_loc is None)
        else:
            loc_ok = (pred_loc is not None and (true_loc.lower() in pred_loc.lower() or pred_loc.lower() in true_loc.lower()))

        results.append({
            "id": sid,
            "spoken_sentence": sc["sentence"],
            "prisma_transcript": transcript,
            "latency_s": lat,
            "true_incident": true_inc,
            "pred_incident": pred_inc,
            "incident_match": int(inc_ok),
            "true_temp": true_temp,
            "pred_temp": pred_temp,
            "temp_match": int(temp_ok),
            "true_dist": true_dist,
            "pred_dist": pred_dist,
            "dist_match": int(dist_ok),
            "true_loc": true_loc,
            "pred_loc": pred_loc,
            "loc_match": int(loc_ok)
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_METRICS_CSV, index=False)
    print(f"\n[+] Itemized metrics saved to {OUTPUT_METRICS_CSV}")

    n = len(df_out)
    inc_acc = df_out["incident_match"].sum() / n * 100.0
    temp_acc = df_out["temp_match"].sum() / n * 100.0
    dist_acc = df_out["dist_match"].sum() / n * 100.0
    loc_acc = df_out["loc_match"].sum() / n * 100.0
    mean_lat = df_out["latency_s"].mean()

    summary_df = pd.DataFrame([{
        "dataset": "Fresh Unseen Evaluation (N=10 Sentences)",
        "incident_accuracy": f"{df_out['incident_match'].sum()}/{n} ({inc_acc:.1f}%)",
        "temperature_accuracy": f"{df_out['temp_match'].sum()}/{n} ({temp_acc:.1f}%)",
        "distance_accuracy": f"{df_out['dist_match'].sum()}/{n} ({dist_acc:.1f}%)",
        "location_accuracy": f"{df_out['loc_match'].sum()}/{n} ({loc_acc:.1f}%)",
        "mean_prisma_latency_s": round(mean_lat, 3)
    }])
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    print(f"[+] Summary saved to {OUTPUT_SUMMARY_CSV}\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
