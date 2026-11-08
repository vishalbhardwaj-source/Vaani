"""
Evaluates Vaani's deterministic extraction engine directly on the 28 live Gnani Prisma ASR transcripts.
Checks real-world end-to-end performance on Devanagari numerals, degree signs, kilometers, and Indic landmarks.
"""

import json
import sys
from pathlib import Path
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gnani.client import extract_incident_details

df_utt = pd.read_csv(PROJECT_ROOT / "benchmark" / "utterances.csv").set_index("id")
raw_prisma = json.load(open(PROJECT_ROOT / "benchmark" / "results" / "raw_live_prisma_responses.json", encoding="utf-8"))

results = []

for clip_key, data in raw_prisma.items():
    # Extract utterance id
    if "telephony" in clip_key:
        uid = clip_key.split("_telephony")[0]
        cond = "telephony" + clip_key.split("_telephony")[1]
    else:
        uid = clip_key.rsplit("_clean", 1)[0]
        cond = "clean"

    transcript = data.get("transcript", "")
    row = df_utt.loc[uid]

    true_inc = str(row["incident_type"]).strip()
    true_temp = row["true_temperature_c"]
    true_dist = row["true_distance_km"]
    true_loc = str(row["true_location"]).strip()

    # Run extraction on actual Prisma output
    rec = extract_incident_details(transcript)

    pred_inc = rec.incident_type.value
    pred_temp = rec.temperature_c
    pred_dist = rec.distance_km
    pred_loc = rec.location_text

    # Scoring
    inc_ok = (pred_inc == true_inc)

    if pd.isna(true_temp):
        temp_ok = (pred_temp is None)
    else:
        temp_ok = (pred_temp is not None and abs(float(pred_temp) - float(true_temp)) < 0.1)

    if pd.isna(true_dist):
        dist_ok = (pred_dist is None)
    else:
        dist_ok = (pred_dist is not None and abs(float(pred_dist) - float(true_dist)) < 0.1)

    if pd.isna(true_loc) or true_loc == "" or true_loc == "None":
        loc_ok = (pred_loc is None)
    else:
        # Check if true location matches or is captured in pred_loc
        if pred_loc:
            loc_ok = (true_loc.lower() in pred_loc.lower() or pred_loc.lower() in true_loc.lower())
        else:
            loc_ok = False

    results.append({
        "clip_key": clip_key,
        "condition": cond,
        "utterance_id": uid,
        "prisma_transcript": transcript,
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

df_res = pd.DataFrame(results)
out_csv = PROJECT_ROOT / "benchmark" / "results" / "prisma_end_to_end_extraction_metrics.csv"
df_res.to_csv(out_csv, index=False)
print(f"Saved itemized metrics to {out_csv}")

# Print summary
total = len(df_res)
inc_acc = df_res["incident_match"].sum() / total * 100.0
temp_acc = df_res["temp_match"].sum() / total * 100.0
dist_acc = df_res["dist_match"].sum() / total * 100.0
loc_acc = df_res["loc_match"].sum() / total * 100.0

print("\n" + "=" * 78)
print("END-TO-END EXTRACTION ACCURACY ON 28 LIVE PRISMA TRANSCRIPTS")
print("=" * 78)
print(f"Total Evaluated Audio Transcripts: {total}")
print(f"Incident Type Accuracy          : {inc_acc:.1f}% ({df_res['incident_match'].sum()}/{total})")
print(f"Temperature Slot Accuracy       : {temp_acc:.1f}% ({df_res['temp_match'].sum()}/{total})")
print(f"Distance Slot Accuracy          : {dist_acc:.1f}% ({df_res['dist_match'].sum()}/{total})")
print(f"Location Landmark Accuracy      : {loc_acc:.1f}% ({df_res['loc_match'].sum()}/{total})")
print("=" * 78)
