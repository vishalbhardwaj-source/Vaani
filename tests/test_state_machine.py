"""
Tests for Vaani Dialogue Controller State Machine.
Verifies the core safety guarantee: Dispatch actions CAN NEVER run without explicit driver confirmation.
"""

import pytest
from pydantic import ValidationError
from core.schemas import DialogueState, ExtractionRecord, DriverReply, IncidentType
from core.state_machine import DialogueController
from core.tools import init_db


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


def test_core_safety_guarantee_dispatch_cannot_run_without_confirmation():
    """
    CRITICAL SAFETY GUARANTEE TEST:
    The language model or controller cannot trigger dispatch tools unless
    explicitly in the ACTING state reached via driver confirmation.
    """
    # DL-01-AA-4092 has target_temp_c = 4.0C (Chiller), so 18C and 28C are within normal plausibility threshold (<= 25C delta)
    controller = DialogueController(vehicle_id="DL-01-AA-4092")
    
    # 1. Test from LISTENING
    assert controller.state == DialogueState.LISTENING
    with pytest.raises(PermissionError) as exc_info:
        controller._execute_dispatch_action()
    assert "SAFETY GATE VIOLATION" in str(exc_info.value)

    # 2. Test from EXTRACTING
    controller.receive_driver_utterance("Bhaiya reefer temperature 18 degree ho gaya")
    assert controller.state == DialogueState.EXTRACTING
    with pytest.raises(PermissionError):
        controller._execute_dispatch_action()

    # 3. Test from CONFIRMING (before reply is received)
    rec = ExtractionRecord(
        incident_type=IncidentType.REEFER_TEMPERATURE,
        temperature_c=18.0,
        location_text="Toll Plaza",
        distance_km=42.0
    )
    controller.process_extraction(rec)
    assert controller.state == DialogueState.CONFIRMING
    
    with pytest.raises(PermissionError):
        controller._execute_dispatch_action()


def test_happy_path_confirmation_to_dispatch():
    """Verify full successful pipeline: LISTENING -> EXTRACTING -> CONFIRMING -> ACTING -> DONE."""
    # DL-01-AA-4092 has target_temp_c = 4.0C (Chiller), so 18C is within normal plausibility threshold (<= 25C delta)
    controller = DialogueController(vehicle_id="DL-01-AA-4092")
    
    # Driver reports
    state, msg = controller.receive_driver_utterance("Reefer temperature 18 degree toll plaza se 42 km aage")
    assert state == DialogueState.EXTRACTING
    
    # Evon extracts
    rec = ExtractionRecord(
        incident_type=IncidentType.REEFER_TEMPERATURE,
        temperature_c=18.0,
        location_text="Toll Plaza",
        distance_km=42.0
    )
    state, prompt = controller.process_extraction(rec)
    assert state == DialogueState.CONFIRMING
    assert "18 degree" in prompt
    assert "42 km" in prompt

    # Driver confirms: "Haan sahi hai"
    reply = DriverReply(intent="CONFIRM", raw_text="haan sahi hai")
    state, spoken_res = controller.handle_confirmation_reply(reply)
    
    assert state == DialogueState.DONE
    assert controller.last_action_result is not None
    assert controller.last_action_result["status"] == "DISPATCHED"
    assert controller.last_action_result["reported_temp_c"] == 18.0


def test_readback_catches_misheard_number_and_corrects():
    """
    Simulate the core paper scenario:
    Prisma mishears 18 as 28.
    Vaani reads back 28.
    Driver rejects and corrects to 18.
    Vaani verifies corrected 18 before dispatching.
    """
    # DL-01-AA-4092 has target_temp_c = 4.0C (Chiller), so 18C and 28C are within normal plausibility threshold (<= 25C delta)
    controller = DialogueController(vehicle_id="DL-01-AA-4092")
    controller.receive_driver_utterance("Bhaiya reefer 18 degree...")

    # Misread extraction (28 instead of 18)
    erroneous_record = ExtractionRecord(
        incident_type=IncidentType.REEFER_TEMPERATURE,
        temperature_c=28.0,
        location_text="Toll Plaza",
        distance_km=42.0
    )
    state, prompt = controller.process_extraction(erroneous_record)
    assert state == DialogueState.CONFIRMING
    assert "28 degree" in prompt

    # Driver catches the mistake: "Nahi, 28 nahi, 18 degree"
    correction_reply = DriverReply(
        intent="CORRECT",
        corrected_temperature_c=18.0,
        raw_text="nahi 28 nahi 18 degree"
    )
    state, updated_prompt = controller.handle_confirmation_reply(correction_reply)
    
    # State must be back in CONFIRMING with corrected 18
    assert state == DialogueState.CONFIRMING
    assert "18 degree" in updated_prompt
    assert controller.current_record.temperature_c == 18.0

    # Driver confirms corrected read-back
    confirm_reply = DriverReply(intent="CONFIRM", raw_text="haan ab sahi hai")
    final_state, action_res = controller.handle_confirmation_reply(confirm_reply)

    assert final_state == DialogueState.DONE
    assert controller.last_action_result["reported_temp_c"] == 18.0
    # Verification: Erroneous 28 was PREVENTED from reaching dispatch!


def test_retry_threshold_escalates_to_human_dispatcher():
    """Verify that repeated confusion (exceeding 2 rounds) safely hands over to human."""
    controller = DialogueController(vehicle_id="HR-55-TC-1102", max_correction_rounds=2)
    controller.receive_driver_utterance("Kharab ho gaya")
    
    rec = ExtractionRecord(
        incident_type=IncidentType.BREAKDOWN,
        location_text="Panipat Highway"
    )
    controller.process_extraction(rec)
    assert controller.state == DialogueState.CONFIRMING

    # Round 1 rejection
    controller.handle_confirmation_reply(DriverReply(intent="REJECT", raw_text="nahi galat hai"))
    assert controller.state == DialogueState.CONFIRMING
    assert controller.correction_rounds == 1

    # Round 2 rejection
    controller.handle_confirmation_reply(DriverReply(intent="REJECT", raw_text="nahi samajh nahi aya"))
    assert controller.state == DialogueState.CONFIRMING
    assert controller.correction_rounds == 2

    # Round 3 -> Exceeds max retries -> ESCALATING
    state, escalate_msg = controller.handle_confirmation_reply(DriverReply(intent="REJECT", raw_text="arre nahi bhai"))
    assert state == DialogueState.ESCALATING
    assert "human fleet supervisor" in escalate_msg.lower()
    assert controller.last_action_result is None  # Dispatch NEVER ran


def test_continuous_learning_records_driver_correction():
    """Verify that driver corrections are persisted into the continuous learning memory store."""
    from core.learning import get_learned_corrections, get_adaptive_rules

    # DL-01-AA-4092 has target_temp_c = 4.0C (Chiller), so 18C and 28C are within normal plausibility threshold (<= 25C delta)
    controller = DialogueController(vehicle_id="DL-01-AA-4092")
    controller.receive_driver_utterance("Reefer 18 degree toll plaza")

    # Erroneous extraction (e.g. 28 instead of 18)
    err_rec = ExtractionRecord(
        incident_type=IncidentType.REEFER_TEMPERATURE,
        temperature_c=28.0,
        location_text="Toll Plaza"
    )
    controller.process_extraction(err_rec)

    # Driver corrects
    reply = DriverReply(
        intent="CORRECT",
        corrected_temperature_c=18.0,
        raw_text="nahi 28 nahi 18 degree hai"
    )
    controller.handle_confirmation_reply(reply)

    # Check learning store
    corrections = get_learned_corrections(limit=5)
    assert len(corrections) > 0
    assert any("18" in str(c["corrected_slots"]) for c in corrections)
    
    # Check adaptive rules updated
    rules = get_adaptive_rules()
    assert len(rules) > 0
    pattern_names = [r["pattern"] for r in rules]
    assert any("18 vs 28" in p or "18 vs 80" in p for p in pattern_names)


def test_pydantic_schema_validation():
    """Verify Pydantic prevents invalid temperatures or distances."""
    with pytest.raises(ValidationError):
        # Impossible temperature for reefer (-80C)
        ExtractionRecord(temperature_c=-80.0, location_text="Depot")

    with pytest.raises(ValidationError):
        # Negative distance
        ExtractionRecord(distance_km=-10.0, location_text="Depot")


def test_committed_benchmark_results_consistency_with_controller():
    """
    REGRESSION TEST:
    Ensures committed benchmark results are strictly up to date with the controller code,
    verifying that the scorecard and controller describe the same system.
    """
    from pathlib import Path
    import pandas as pd

    repo_root = Path(__file__).resolve().parent.parent
    controller_path = repo_root / "core" / "state_machine.py"
    raw_csv_path = repo_root / "benchmark" / "results" / "raw_benchmark_results.csv"
    summary_csv_path = repo_root / "benchmark" / "results" / "summary_metrics.csv"
    sweep_csv_path = repo_root / "benchmark" / "results" / "driver_sensitivity_sweep.csv"
    meta_json_path = repo_root / "benchmark" / "results" / "benchmark_meta.json"

    assert raw_csv_path.exists(), "raw_benchmark_results.csv missing!"
    assert summary_csv_path.exists(), "summary_metrics.csv missing!"
    assert sweep_csv_path.exists(), "driver_sensitivity_sweep.csv missing!"
    assert meta_json_path.exists(), "benchmark_meta.json missing!"

    # 1. Cryptographic Hash Guardrail: results must strictly match active benchmark dependencies
    # (covers state machine, schemas, tools, learning, extractor client, runner, scorer, and utterances.csv)
    import json
    from core.state_machine import get_controller_code_hash
    current_code_hash = get_controller_code_hash()

    summary_df = pd.read_csv(summary_csv_path)
    assert "Controller Code Hash" in summary_df.columns, "summary_metrics.csv missing 'Controller Code Hash' column!"
    committed_hash = str(summary_df["Controller Code Hash"].iloc[0]).strip()

    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    meta_hash = meta_data.get("benchmark_dependencies_hash", "")

    assert committed_hash == current_code_hash, (
        f"Committed benchmark results hash ({committed_hash}) does not match current source hash ({current_code_hash})! "
        "Benchmark code or utterances.csv has changed. Run python benchmark/run_benchmark.py to synchronize."
    )
    assert meta_hash == current_code_hash, (
        f"benchmark_meta.json hash ({meta_hash}) does not match current source hash ({current_code_hash})!"
    )

    # 2. Results content integrity
    df = pd.read_csv(raw_csv_path)
    assert len(df) == 100, f"Expected 100 benchmark rows, got {len(df)}"
    assert "telematics_challenged" in df.columns
    assert "telematics_blocked" in df.columns

    # 3. Safety consistency: blocked telematics anomalies CAN NEVER reach DONE or execute actions
    blocked_rows = df[df["telematics_blocked"] == True]
    for _, row in blocked_rows.iterrows():
        assert row["action_executed"] == False, f"Row {row['utt_id']} had action_executed=True despite gate block!"
        assert row["final_state"] != "DONE", f"Row {row['utt_id']} reached DONE despite gate block!"
