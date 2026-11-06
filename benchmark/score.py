"""
Vaani: Empirical Benchmark Scoring Module
Computes:
1. Word Error Rate (WER) using jiwer
2. Raw Critical Slot Extraction Accuracy (before human read-back)
3. Errors Caught by Spoken Read-Back Gate
4. Silent Failures (Uncaught errors slipping through to dispatch)
5. True Final Dispatch Accuracy (Non-circular calculation)
6. Parametric Driver Sensitivity Sweep (Evaluating silent failure rate under varying driver attention: 0% to 20%)
7. Engine Breakdown (Cloud Prisma, Local Whisper-LoRA, Acoustic Simulation)
"""

import re
from pathlib import Path
import pandas as pd
import jiwer

RAW_RESULTS_CSV = Path(__file__).resolve().parent / "results" / "raw_benchmark_results.csv"
SUMMARY_CSV = Path(__file__).resolve().parent / "results" / "summary_metrics.csv"
SWEEP_CSV = Path(__file__).resolve().parent / "results" / "driver_sensitivity_sweep.csv"


def clean_text_for_wer(text: str) -> str:
    """Normalize text for consistent WER scoring."""
    t = str(text).lower()
    t = re.sub(r"[^\w\s]", "", t)
    return " ".join(t.split())


def compute_metrics() -> pd.DataFrame:
    if not RAW_RESULTS_CSV.exists():
        raise FileNotFoundError(f"Raw results not found at: {RAW_RESULTS_CSV}")

    df = pd.read_csv(RAW_RESULTS_CSV)

    summary_rows = []
    conditions = ["clean", "telephony_8khz", "telephony_10db", "telephony_5db"]

    print("=" * 85)
    print("VAANI EMPIRICAL BENCHMARK EVALUATION METRICS")
    print("=" * 85)

    for cond in conditions:
        sub = df[df["condition"] == cond]
        n_samples = len(sub)
        if n_samples == 0:
            continue

        # 1. Word Error Rate (WER)
        ref_texts = [clean_text_for_wer(t) for t in sub["reference_text"]]
        hyp_texts = [clean_text_for_wer(t) for t in sub["transcript"]]
        wer_val = jiwer.wer(ref_texts, hyp_texts) * 100.0

        # 2. Raw Critical Slot Accuracy (Before Read-Back)
        total_errors = int(sub["has_extraction_error"].sum())
        correct_extractions = n_samples - total_errors
        slot_acc_raw = (correct_extractions / n_samples) * 100.0

        # 3. Errors Caught by Read-Back
        errors_caught = int(sub[sub["has_extraction_error"]]["error_caught"].sum())
        caught_pct = (errors_caught / total_errors * 100.0) if total_errors > 0 else 100.0

        # 4. Silent Failures (Driver misheard / reflexively confirmed incorrect details)
        silent_fails = int(sub["silent_failure"].sum())
        silent_fail_pct = (silent_fails / n_samples) * 100.0

        # 5. Non-Circular True Dispatch Accuracy
        true_dispatch_acc = ((n_samples - silent_fails) / n_samples) * 100.0

        # Engine source primary
        sources = sub["source"].unique() if "source" in sub.columns else ["Simulation"]
        source_label = sources[0] if len(sources) == 1 else "Mixed"

        summary_rows.append({
            "Condition": cond,
            "Sample Size (N)": n_samples,
            "Source Engine": source_label,
            "WER (%)": round(wer_val, 2),
            "Raw Slot Acc (%)": round(slot_acc_raw, 2),
            "Total Errors": total_errors,
            "Errors Caught": f"{errors_caught} of {total_errors} ({round(caught_pct, 1)}%)",
            "Silent Failures": f"{silent_fails} ({round(silent_fail_pct, 1)}%)",
            "Final Dispatch Accuracy (%)": round(true_dispatch_acc, 2)
        })

    summary_df = pd.DataFrame(summary_rows)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print(summary_df.to_string(index=False))
    print("=" * 85)
    print(f"Summary metrics saved to: {SUMMARY_CSV}")

    # Compute Sensitivity Sweep
    compute_sensitivity_sweep(df)

    return summary_df


def compute_sensitivity_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parametric sensitivity analysis:
    Evaluates final dispatch accuracy across assumed driver reflexive error rates (0% to 20%).
    Shows how dispatch integrity varies as a function of driver attentiveness and cabin noise.
    """
    sweep_rates = [0.0, 0.05, 0.10, 0.15, 0.20]
    sweep_rows = []
    conditions = ["clean", "telephony_8khz", "telephony_10db", "telephony_5db"]

    print("\n" + "=" * 85)
    print("PARAMETRIC DRIVER ATTENTION SENSITIVITY SWEEP (0% to 20% Reflexive Agreement)")
    print("=" * 85)

    for rate in sweep_rates:
        for cond in conditions:
            sub = df[df["condition"] == cond]
            total_clips = len(sub)
            if total_clips == 0:
                continue

            extraction_errors = int(sub["has_extraction_error"].sum())
            correct_raw = total_clips - extraction_errors

            # Model expected silent failures under this driver error rate
            expected_silent_fails = round(extraction_errors * rate, 2)
            expected_caught = round(extraction_errors * (1.0 - rate), 2)
            dispatch_acc = round(((total_clips - expected_silent_fails) / total_clips) * 100.0, 1)

            sweep_rows.append({
                "Assumed Driver Error Rate (%)": f"{int(rate * 100)}%",
                "Condition": cond,
                "Raw Slot Acc (%)": round((correct_raw / total_clips) * 100.0, 1),
                "Total ASR Errors": extraction_errors,
                "Expected Caught": expected_caught,
                "Expected Silent Failures": expected_silent_fails,
                "Modeled Dispatch Accuracy (%)": dispatch_acc
            })

    sweep_df = pd.DataFrame(sweep_rows)
    SWEEP_CSV.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(SWEEP_CSV, index=False)
    print(sweep_df.to_string(index=False))
    print(f"\nDriver sensitivity sweep saved to: {SWEEP_CSV}")
    return sweep_df


if __name__ == "__main__":
    compute_metrics()
