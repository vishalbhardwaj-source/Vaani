"""
Vaani: Empirical End-to-End Held-Out Slot Accuracy Evaluation
Pipeline: Audio -> Whisper-LoRA ASR -> Evon/Extractor -> Critical Slots
Evaluates on all 28 held-out audio clips (7 unseen test utterances x 4 acoustic conditions).
"""

import sys
from pathlib import Path
import pandas as pd
import soundfile as sf
import scipy.signal as signal
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.asr_model import get_asr_service
from gnani.client import extract_incident_details
UTTERANCES_CSV = PROJECT_ROOT / "benchmark" / "utterances.csv"
OUT_CSV = PROJECT_ROOT / "benchmark" / "results" / "held_out_e2e_slot_metrics.csv"
HELD_OUT_IDS = ["utt_03", "utt_07", "utt_11", "utt_15", "utt_19", "utt_22", "utt_25"]
CONDITIONS = ["clean", "telephony_8khz", "telephony_10db", "telephony_5db"]


def run_e2e_evaluation() -> pd.DataFrame:
    print("=" * 80)
    print("EVALUATING END-TO-END HELD-OUT PIPELINE (AUDIO -> ASR -> EXTRACTOR -> SLOTS)")
    print("=" * 80)

    asr = get_asr_service()
    proc = asr.processor
    model = asr.model

    df_utt = pd.read_csv(UTTERANCES_CSV)
    df_held = df_utt[df_utt["id"].isin(HELD_OUT_IDS)]

    results = []

    for cond in CONDITIONS:
        tot_slots = 0
        cor_slots = 0
        incident_cor = 0
        temp_cor = 0
        dist_cor = 0
        loc_cor = 0

        for _, row in df_held.iterrows():
            uid = row["id"]
            wav_path = PROJECT_ROOT / "benchmark" / "audio" / f"{uid}_{cond}.wav"
            data, sr = sf.read(str(wav_path))
            if sr != 16000:
                audio_16k = signal.resample_poly(data, 16000, sr)
            else:
                audio_16k = data

            in_t = proc(audio_16k, sampling_rate=16000, return_tensors="pt")
            in_t = {k: v.to("cpu") for k, v in in_t.items()}
            with torch.no_grad():
                gen = model.generate(**in_t, max_new_tokens=128, language="en", task="transcribe")
                hyp = proc.batch_decode(gen, skip_special_tokens=True)[0].strip()

            rec = extract_incident_details(hyp)

            # 1. Incident Type Slot
            tot_slots += 1
            if rec.incident_type.value == str(row["incident_type"]).lower():
                cor_slots += 1
                incident_cor += 1

            # 2. Temperature Slot
            exp_temp = row["true_temperature_c"]
            tot_slots += 1
            if pd.isna(exp_temp):
                if rec.temperature_c is None:
                    cor_slots += 1
                    temp_cor += 1
            else:
                if rec.temperature_c is not None and abs(rec.temperature_c - float(exp_temp)) < 0.1:
                    cor_slots += 1
                    temp_cor += 1

            # 3. Distance Slot
            exp_dist = row["true_distance_km"]
            tot_slots += 1
            if pd.isna(exp_dist):
                if rec.distance_km is None:
                    cor_slots += 1
                    dist_cor += 1
            else:
                if rec.distance_km is not None and abs(rec.distance_km - float(exp_dist)) <= 0.5:
                    cor_slots += 1
                    dist_cor += 1

            # 4. Location Slot
            exp_loc = row["true_location"]
            tot_slots += 1
            if pd.isna(exp_loc):
                if rec.location_text is None:
                    cor_slots += 1
                    loc_cor += 1
            else:
                if rec.location_text and (str(exp_loc).lower() in rec.location_text.lower() or rec.location_text.lower() in str(exp_loc).lower()):
                    cor_slots += 1
                    loc_cor += 1

        acc_pct = round((cor_slots / tot_slots) * 100.0, 1)
        results.append({
            "Condition": cond,
            "Held-Out Clips (N)": len(df_held),
            "Total Evaluated Slots": tot_slots,
            "Correct Slots": cor_slots,
            "End-to-End Slot Accuracy": f"{cor_slots}/{tot_slots} ({acc_pct}%)",
            "Incident Type Acc": f"{incident_cor}/{len(df_held)}",
            "Temperature Slot Acc": f"{temp_cor}/{len(df_held)}",
            "Distance Slot Acc": f"{dist_cor}/{len(df_held)}",
            "Location Slot Acc": f"{loc_cor}/{len(df_held)}"
        })

    out_df = pd.DataFrame(results)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print("\nHELD-OUT END-TO-END PIPELINE METRICS:")
    print(out_df.to_string(index=False))
    print(f"\nSaved held-out E2E metrics to: {OUT_CSV}")
    return out_df


if __name__ == "__main__":
    run_e2e_evaluation()
