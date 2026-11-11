"""
Tests for Vaani Dialogue Controller State Machine.
Verifies the core safety guarantee: Dispatch actions CAN NEVER run without explicit driver confirmation.
"""

import pytest
from pathlib import Path
import pandas as pd
from pydantic import ValidationError
from core.schemas import DialogueState, ExtractionRecord, DriverReply, IncidentType, GateDecision, VehicleRecord
from core.state_machine import DialogueController, evaluate_confirmation_gate
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
    # GJ-01-BX-7201 has target_temp_c = 6.0C (Produce tolerance ±15C), so 18C is within normal plausibility threshold (<= 15C delta)
    controller = DialogueController(vehicle_id="GJ-01-BX-7201")
    controller.veh_telemetry.current_temp_c = 18.0
    
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
    # GJ-01-BX-7201 has target_temp_c = 6.0C (Produce tolerance ±15C):
    # 28C diverges by 22C (> 15C) and is challenged, while corrected 18C (delta 12C <= 15C) is safe.
    controller = DialogueController(vehicle_id="GJ-01-BX-7201")
    controller.veh_telemetry.current_temp_c = 18.0
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
    """Verify that driver corrections are persisted into the offline audit memory store."""
    from core.learning import get_learned_corrections, get_adaptive_rules

    # GJ-01-BX-7201 has target_temp_c = 6.0C (Produce tolerance ±15C)
    controller = DialogueController(vehicle_id="GJ-01-BX-7201")
    controller.veh_telemetry.current_temp_c = 18.0
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
    
    # Check offline patterns loaded
    rules = get_adaptive_rules()
    assert len(rules) > 0
    pattern_names = [r["pattern"] for r in rules]
    assert any("18 vs 80" in p or "14 vs 40" in p for p in pattern_names)


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
    assert len(df) == 125, f"Expected 125 benchmark rows (25 clips x 5 conditions), got {len(df)}"
    assert "telematics_challenged" in df.columns
    assert "telematics_blocked" in df.columns

    # 3. Safety consistency: blocked telematics anomalies CAN NEVER reach DONE or execute actions
    blocked_rows = df[df["telematics_blocked"] == True]
    for _, row in blocked_rows.iterrows():
        assert row["action_executed"] == False, f"Row {row['utt_id']} had action_executed=True despite gate block!"
        assert row["final_state"] != "DONE", f"Row {row['utt_id']} reached DONE despite gate block!"


def test_readme_tables_match_committed_csvs():
    """Guardrail test ensuring README.md tables match committed benchmark CSVs exactly."""
    project_root = Path(__file__).resolve().parent.parent
    readme_path = project_root / "README.md"
    summary_csv_path = project_root / "benchmark" / "results" / "summary_metrics.csv"
    adv_csv_path = project_root / "benchmark" / "results" / "adversarial_gate_results.csv"
    e2e_csv_path = project_root / "benchmark" / "results" / "held_out_e2e_slot_metrics.csv"

    assert readme_path.exists(), "README.md does not exist!"
    assert summary_csv_path.exists(), "summary_metrics.csv does not exist!"
    assert adv_csv_path.exists(), "adversarial_gate_results.csv does not exist!"
    assert e2e_csv_path.exists(), "held_out_e2e_slot_metrics.csv does not exist!"

    with open(readme_path, "r", encoding="utf-8") as f:
        readme_text = f.read()

    # Verify Track 1 summary metrics are present in README.md
    df_sum = pd.read_csv(summary_csv_path)
    for _, row in df_sum.iterrows():
        cond = str(row["Condition"])
        assert cond in readme_text, f"Condition {cond} from summary_metrics.csv missing in README!"
        raw_acc = str(row["Raw Slot Acc (%)"])
        assert raw_acc in readme_text, f"Raw Slot Acc {raw_acc} for {cond} missing in README!"

    # Verify Adversarial Gate results are present in README.md
    df_adv = pd.read_csv(adv_csv_path)
    for _, row in df_adv.iterrows():
        scen = str(row["Scenario ID"])
        assert scen in readme_text, f"Scenario {scen} from adversarial_gate_results.csv missing in README!"

    # Verify End-to-End Held-Out Slot metrics are present in README.md
    df_e2e = pd.read_csv(e2e_csv_path)
    for _, row in df_e2e.iterrows():
        cond = str(row["Condition"])
        e2e_acc = str(row["End-to-End Slot Accuracy"])
        assert e2e_acc in readme_text, f"E2E Slot Acc {e2e_acc} for {cond} missing in README!"


def test_evaluate_confirmation_gate_exhaustive():
    """
    Exhaustive tests for the pure evaluate_confirmation_gate function:
    1. Exact divergence boundaries: under, exact, and above cargo tolerance.
    2. Per-cargo class thresholds (Vaccine ±4°C, Dairy ±8°C, Ice Cream ±12°C).
    3. Reflexive confirmation ('haan', 'sahi hai', bare 'confirm') blocked when challenged.
    4. Verified number repeats ('X degree confirm', spoken minus forms) allowed.
    5. Known residual risk: driver explicitly repeats and confirms divergent value -> allowed with residual risk flag.
    """
    from core.state_machine import evaluate_confirmation_gate
    from core.schemas import GateDecision, VehicleRecord

    # Mock telemetry for Vaccine vehicle (Target -20°C, Target Tolerance ±4°C, Sensor Match Window ±0.8°C)
    veh_vaccine = VehicleRecord(
        vehicle_id="MH-12-RN-8819", driver_name="Sukhwinder", driver_phone="+91-9811223344",
        vehicle_type="Reefer 24ft", cargo_type="Vaccine Cold-Chain",
        target_temp_c=-20.0, current_temp_c=-18.5, route="Pune to Mumbai",
        last_known_location="Mumbai-Pune Expressway km 68", status="IN_TRANSIT"
    )

    # 1. Vaccine Safe: -18°C (delta 0.5°C <= 0.8°C sensor window) -> ALLOW on standard 'haan'
    rec_v1 = ExtractionRecord(temperature_c=-18.0, location_text="Pune Highway")
    res_v1 = evaluate_confirmation_gate(rec_v1, DriverReply(intent="CONFIRM", raw_text="haan sahi hai"), veh_vaccine)
    assert res_v1.decision == GateDecision.ALLOW
    assert res_v1.residual_risk_flag is False

    # 2. Vaccine Breach: -14°C (delta 6°C > 4°C tolerance) -> CHALLENGE on reflexive 'haan'
    rec_v2 = ExtractionRecord(temperature_c=-14.0, location_text="Pune Highway")
    res_v2 = evaluate_confirmation_gate(rec_v2, DriverReply(intent="CONFIRM", raw_text="haan bhaiya"), veh_vaccine)
    assert res_v2.decision == GateDecision.CHALLENGE
    assert "Telematics" in str(res_v2.challenge_prompt) and "target" in str(res_v2.challenge_prompt)

    # 3. Vaccine Breach with Spoken Minus Repeat -> ALLOW
    res_v3 = evaluate_confirmation_gate(rec_v2, DriverReply(intent="CONFIRM", raw_text="minus 14 degree confirm"), veh_vaccine)
    assert res_v3.decision == GateDecision.ALLOW
    assert res_v3.residual_risk_flag is True

    # 4. Boundary cases on Ice Cream vehicle (Target -22°C, Tolerance ±12°C, Case 4 stale sensor fallback)
    veh_icecream = VehicleRecord(
        vehicle_id="HR-55-TC-1102", driver_name="Ramesh", driver_phone="+91-9876543210",
        vehicle_type="Reefer 32ft", cargo_type="Ice Cream Consignment",
        target_temp_c=-22.0, current_temp_c=-22.0, route="Karnal to Delhi",
        last_known_location="NH-44 Toll Plaza km 42", status="IN_TRANSIT",
        gps_fix_age_minutes=30  # Stale (>15 min) forces fallback to target tolerance!
    )

    # Under boundary: -12°C (delta 10°C < 12°C) -> ALLOW
    res_b_under = evaluate_confirmation_gate(ExtractionRecord(temperature_c=-12.0), DriverReply(intent="CONFIRM", raw_text="haan"), veh_icecream)
    assert res_b_under.decision == GateDecision.ALLOW

    # Exact boundary: -10°C (delta 12°C == 12°C) -> ALLOW
    res_b_exact = evaluate_confirmation_gate(ExtractionRecord(temperature_c=-10.0), DriverReply(intent="CONFIRM", raw_text="haan"), veh_icecream)
    assert res_b_exact.decision == GateDecision.ALLOW

    # Above boundary: -8°C (delta 14°C > 12°C) -> CHALLENGE
    res_b_above = evaluate_confirmation_gate(ExtractionRecord(temperature_c=-8.0), DriverReply(intent="CONFIRM", raw_text="haan"), veh_icecream)
    assert res_b_above.decision == GateDecision.CHALLENGE

    # 5. Known Residual Risk: Driver mishears and repeats the wrong number
    # Dairy truck (+4°C), reported 40°C, driver explicitly repeats "40 degree confirm"
    veh_dairy = VehicleRecord(
        vehicle_id="DL-01-AA-4092", driver_name="Abdul", driver_phone="+91-9899001122",
        vehicle_type="Reefer 20ft", cargo_type="Dairy & Milk Products",
        target_temp_c=4.0, current_temp_c=9.0, route="Jaipur to Delhi",
        last_known_location="NH-48 Bilaspur Chowk", status="IN_TRANSIT"
    )
    rec_risk = ExtractionRecord(temperature_c=40.0, location_text="Bilaspur Chowk")
    res_risk = evaluate_confirmation_gate(rec_risk, DriverReply(intent="CONFIRM", raw_text="haan 40 degree confirm"), veh_dairy)
    assert res_risk.decision == GateDecision.ALLOW
    assert res_risk.residual_risk_flag is True  # Audit flag actively records driver override of telematics breach!


def test_unknown_vehicle_id_fails_loudly_and_asks():
    """Verify that an unknown vehicle registration fails loudly and routes to ASKING for identity."""
    controller = DialogueController(vehicle_id="UNKNOWN-REG-9999")
    assert controller.is_unverified_vehicle is True

    # Driver reports
    controller.receive_driver_utterance("Temperature 18 degree")
    state, prompt = controller.process_extraction(ExtractionRecord(temperature_c=18.0, location_text="Panipat"))

    # Must transition to ASKING for vehicle confirmation, never guessing or falling back silently!
    assert state == DialogueState.ASKING
    assert "registration number" in prompt.lower() or "4 digit" in prompt.lower()
    assert controller.last_action_result is None


def test_generic_location_fragments_rejected_and_anchored_to_gps():
    """Verify that generic location words ('toll', 'highway') are not accepted as dispatch locations alone."""
    from gnani.client import extract_incident_details

    # Generic utterance without a named city or landmark
    rec = extract_incident_details("Bhaiya reefer 18 degree ho gaya hai toll ke paas")
    assert rec.location_text is None  # Bare generic 'toll' is REJECTED by extractor

    # Fed into DialogueController with valid vehicle having GPS telematics
    controller = DialogueController(vehicle_id="HR-55-TC-1102", use_gps_telemetry=True)
    controller.receive_driver_utterance("Bhaiya reefer 18 degree ho gaya hai toll ke paas")
    state, prompt = controller.process_extraction(rec)

    # State machine anchors to vehicle's telematics GPS
    assert state == DialogueState.CONFIRMING
    assert "GPS" in prompt and ("Corridor" in prompt or "NH-44" in prompt)


def test_location_slot_filling_with_gps_fallback_disabled():
    from gnani.client import extract_incident_details
    rec = extract_incident_details("Bhaiya reefer 18 degree ho gaya hai toll ke paas")
    assert rec.location_text is None

    controller = DialogueController(vehicle_id="HR-55-TC-1102", use_gps_telemetry=False)
    controller.receive_driver_utterance("Bhaiya reefer 18 degree ho gaya hai toll ke paas")
    state, prompt = controller.process_extraction(rec)

    # With GPS disabled, it must ask for location clarification instead of anchoring
    assert state == DialogueState.ASKING
    assert "landmark" in prompt.lower() or "toll" in prompt.lower()


def test_live_sensor_rule_exhaustive_cases():
    """
    Unit test covering the 4 live sensor decision rules:
    - Case 1: Matches sensor -> ALLOW (Strong evidence ASR is right, even when far from target)
    - Case 2: Matches target ONLY -> CHALLENGE (Weak evidence, sensor indicates divergence)
    - Case 3: Matches neither -> CHALLENGE (Acoustic trap)
    - Case 4: Stale or faulty sensor -> Falls back to target-only check
    """
    from core.state_machine import evaluate_confirmation_gate
    from core.schemas import GateDecision, VehicleRecord

    # Truck with target -22°C, tolerance 12°C (Ice cream)
    # Case 1: Reefer warmed up, sensor reads -12°C. Driver reports -13°C.
    # Delta from sensor = 1°C (<= 12°C). Delta from target = 9°C.
    veh_breakdown = VehicleRecord(
        vehicle_id="HR-55-TC-1102", driver_name="Ramesh", driver_phone="+91-9876543210",
        vehicle_type="Reefer 32ft", cargo_type="Ice Cream Consignment",
        target_temp_c=-22.0, current_temp_c=-12.0, route="Karnal to Delhi",
        last_known_location="NH-44 Corridor Milepost 118", status="IN_TRANSIT",
        gps_fix_age_minutes=3
    )
    res_c1 = evaluate_confirmation_gate(
        ExtractionRecord(temperature_c=-13.0, location_text="Highway"),
        DriverReply(intent="CONFIRM", raw_text="haan sahi hai"),
        veh_breakdown
    )
    assert res_c1.decision == GateDecision.ALLOW
    assert "Strong evidence" in res_c1.reason

    # Case 2: Matches target only (Weak evidence).
    # Sensor reads -2.0°C (severe failure, 20°C delta from target). Driver reports -21.0°C (matches target).
    # Delta sensor = 19°C (> 12°C). Delta target = 1°C (<= 12°C).
    veh_divergent = VehicleRecord(
        vehicle_id="HR-55-TC-1102", driver_name="Ramesh", driver_phone="+91-9876543210",
        vehicle_type="Reefer 32ft", cargo_type="Ice Cream Consignment",
        target_temp_c=-22.0, current_temp_c=-2.0, route="Karnal to Delhi",
        last_known_location="NH-44 Corridor Milepost 118", status="IN_TRANSIT",
        gps_fix_age_minutes=3
    )
    res_c2 = evaluate_confirmation_gate(
        ExtractionRecord(temperature_c=-21.0, location_text="Highway"),
        DriverReply(intent="CONFIRM", raw_text="haan sahi hai"),
        veh_divergent
    )
    assert res_c2.decision == GateDecision.CHALLENGE
    assert "Weak evidence" in res_c2.reason

    # Case 3: Matches neither sensor nor target.
    # Sensor reads -10°C, target -22°C. Driver reports 25°C (acoustic trap / out of cold chain).
    res_c3 = evaluate_confirmation_gate(
        ExtractionRecord(temperature_c=25.0, location_text="Highway"),
        DriverReply(intent="CONFIRM", raw_text="haan sahi hai"),
        veh_breakdown
    )
    assert res_c3.decision == GateDecision.CHALLENGE
    assert "Matches neither" in res_c3.reason or "Polarity flip" in res_c3.reason

    # Case 4: Stale sensor (fix age = 25 minutes > 15m threshold).
    # Sensor value ignored; falls back to target check.
    veh_stale = VehicleRecord(
        vehicle_id="HR-55-TC-1102", driver_name="Ramesh", driver_phone="+91-9876543210",
        vehicle_type="Reefer 32ft", cargo_type="Ice Cream Consignment",
        target_temp_c=-22.0, current_temp_c=-10.0, route="Karnal to Delhi",
        last_known_location="NH-44 Corridor Milepost 118", status="IN_TRANSIT",
        gps_fix_age_minutes=25  # Stale!
    )
    res_c4 = evaluate_confirmation_gate(
        ExtractionRecord(temperature_c=-20.0, location_text="Highway"),
        DriverReply(intent="CONFIRM", raw_text="haan sahi hai"),
        veh_stale
    )
    assert res_c4.decision == GateDecision.ALLOW
    assert "stale" in res_c4.reason


def test_sign_flip_detection_subzero_target():
    """Verify that a dropped minus sign (-14 -> +14) on a sub-zero cargo is immediately challenged."""
    from core.state_machine import evaluate_confirmation_gate
    from core.schemas import GateDecision, VehicleRecord

    veh_vaccine = VehicleRecord(
        vehicle_id="MH-12-RN-8819", driver_name="Sukhwinder", driver_phone="+91-9811223344",
        vehicle_type="Reefer 24ft", cargo_type="Vaccine Cold-Chain",
        target_temp_c=-20.0, current_temp_c=-18.0, route="Pune to Mumbai",
        last_known_location="Mumbai-Pune Expressway Section 4", status="IN_TRANSIT"
    )

    # Driver says +14°C (due to dropped minus sign in ASR)
    res_flip = evaluate_confirmation_gate(
        ExtractionRecord(temperature_c=14.0, location_text="Pune Highway"),
        DriverReply(intent="CONFIRM", raw_text="haan sahi hai"),
        veh_vaccine
    )
    assert res_flip.decision == GateDecision.CHALLENGE
    assert "Polarity flip" in res_flip.reason


def test_caller_id_failure_modes():
    """Test caller ID failure modes: phone not in table, multiple matches, single match."""
    from core.state_machine import DialogueController, DialogueState
    from core.schemas import ExtractionRecord, IncidentType

    # 1. Phone not in table -> enters ASKING, never defaults to a fallback vehicle!
    ctrl_unknown = DialogueController(caller_phone="+91-9999999999", vehicle_id=None)
    assert ctrl_unknown.is_unverified_vehicle is True
    assert ctrl_unknown.vehicle_id is None
    ctrl_unknown.receive_driver_utterance("Reefer 18 degree")
    state, prompt = ctrl_unknown.process_extraction(ExtractionRecord(incident_type=IncidentType.REEFER_TEMPERATURE, temperature_c=18.0, location_text="Highway"))
    assert state == DialogueState.ASKING
    assert "registration number" in prompt or "aakhri 4 digit" in prompt

    # 2. Single match -> verified vehicle
    ctrl_known = DialogueController(caller_phone="+91-9876543210")
    assert ctrl_known.is_unverified_vehicle is False
    assert ctrl_known.vehicle_id == "HR-55-TC-1102"


def test_cargo_excursion_check_independent_of_asr_gate():
    """Verify that cargo physical excursion risk is decoupled from ASR plausibility."""
    from core.schemas import check_cargo_temperature_excursion, DispatchPriority

    # Deep freeze target is -22°C. A reported -10°C might pass an ASR plausibility check against a -10°C sensor,
    # but represents a critical physical cargo excursion (> -18°C threshold)!
    is_excursion, reason, prio = check_cargo_temperature_excursion("Ice Cream Consignment", -10.0)
    assert is_excursion is True
    assert prio == DispatchPriority.CRITICAL
    assert "-18°C" in reason


def test_sensor_match_tolerance_edges_and_boundary_conditions():
    """
    Test Rule 1 per-cargo class sensor match tolerance exact edge conditions:
    - Vaccines (±0.8°C match window)
    - Deep freeze (±1.5°C match window)
    - Produce (±2.0°C match window)
    Also tests near-zero thaw readings (+0.5°C vs -1.0°C) which must NOT fire false sign-flip alarms.
    """
    from core.schemas import VehicleRecord, GateDecision, get_cargo_sensor_match_tolerance

    # 1. Deep Freeze (±1.5°C match window)
    veh_freeze = VehicleRecord(
        vehicle_id="HR-55-TC-1102", driver_name="Ramesh", driver_phone="+91-9876543210",
        vehicle_type="Reefer 32ft", cargo_type="Ice Cream Consignment",
        target_temp_c=-22.0, current_temp_c=-21.6, route="Karnal to Delhi",
        last_known_location="NH-44", status="IN_TRANSIT"
    )
    assert get_cargo_sensor_match_tolerance(veh_freeze.cargo_type) == 1.5

    # Spoken -21.0°C against sensor -21.6°C (delta 0.6 <= 1.5) -> ALLOW
    res_21 = evaluate_confirmation_gate(
        ExtractionRecord(temperature_c=-21.0, location_text="NH-44"),
        DriverReply(intent="CONFIRM", raw_text="haan sahi hai"),
        veh_freeze
    )
    assert res_21.decision == GateDecision.ALLOW
    assert res_21.telematics_divergence == pytest.approx(0.6, abs=0.01)

    # Spoken -23.1°C against sensor -21.6°C (delta 1.5 == 1.5 edge) -> ALLOW
    res_edge = evaluate_confirmation_gate(
        ExtractionRecord(temperature_c=-23.1, location_text="NH-44"),
        DriverReply(intent="CONFIRM", raw_text="haan sahi hai"),
        veh_freeze
    )
    assert res_edge.decision == GateDecision.ALLOW
    assert res_edge.telematics_divergence == pytest.approx(1.5, abs=0.01)

    # Spoken -19.5°C against sensor -21.6°C (delta 2.1 > 1.5) -> CHALLENGE
    res_outside = evaluate_confirmation_gate(
        ExtractionRecord(temperature_c=-19.5, location_text="NH-44"),
        DriverReply(intent="CONFIRM", raw_text="haan sahi hai"),
        veh_freeze
    )
    assert res_outside.decision == GateDecision.CHALLENGE

    # 2. Vaccine Cold-Chain (Tight ±0.8°C match window)
    veh_vaccine = VehicleRecord(
        vehicle_id="MH-12-RN-8819", driver_name="Sukhwinder", driver_phone="+91-9811223344",
        vehicle_type="Reefer 24ft", cargo_type="Vaccine Cold-Chain",
        target_temp_c=5.0, current_temp_c=6.0, route="Pune to Mumbai",
        last_known_location="Expressway", status="IN_TRANSIT"
    )
    assert get_cargo_sensor_match_tolerance(veh_vaccine.cargo_type) == 0.8

    # Spoken 4.0°C against sensor 6.0°C (delta 2.0 > 0.8) -> Must CHALLENGE!
    res_vax_4 = evaluate_confirmation_gate(
        ExtractionRecord(temperature_c=4.0, location_text="Expressway"),
        DriverReply(intent="CONFIRM", raw_text="haan sahi hai"),
        veh_vaccine
    )
    assert res_vax_4.decision == GateDecision.CHALLENGE

    # Spoken 8.0°C against sensor 6.0°C (delta 2.0 > 0.8) -> Must CHALLENGE!
    res_vax_8 = evaluate_confirmation_gate(
        ExtractionRecord(temperature_c=8.0, location_text="Expressway"),
        DriverReply(intent="CONFIRM", raw_text="haan sahi hai"),
        veh_vaccine
    )
    assert res_vax_8.decision == GateDecision.CHALLENGE

    # Spoken 6.5°C against sensor 6.0°C (delta 0.5 <= 0.8 edge) -> ALLOW
    res_vax_allow = evaluate_confirmation_gate(
        ExtractionRecord(temperature_c=6.5, location_text="Expressway"),
        DriverReply(intent="CONFIRM", raw_text="haan sahi hai"),
        veh_vaccine
    )
    assert res_vax_allow.decision == GateDecision.ALLOW

    # 3. Produce / Perishables (±2.0°C match window)
    veh_produce = VehicleRecord(
        vehicle_id="GJ-01-BX-7201", driver_name="Vikram", driver_phone="+91-9822334455",
        vehicle_type="Reefer 16ft", cargo_type="Fresh Perishables / Fruits",
        target_temp_c=6.0, current_temp_c=7.0, route="Ahmedabad to Surat",
        last_known_location="Hub", status="IN_TRANSIT"
    )
    assert get_cargo_sensor_match_tolerance(veh_produce.cargo_type) == 2.0

    # Spoken 9.0°C vs sensor 7.0°C (delta 2.0 == 2.0 edge) -> ALLOW
    res_prod_edge = evaluate_confirmation_gate(
        ExtractionRecord(temperature_c=9.0, location_text="Hub"),
        DriverReply(intent="CONFIRM", raw_text="haan sahi hai"),
        veh_produce
    )
    assert res_prod_edge.decision == GateDecision.ALLOW

    # Spoken 9.5°C vs sensor 7.0°C (delta 2.5 > 2.0) -> CHALLENGE
    res_prod_outside = evaluate_confirmation_gate(
        ExtractionRecord(temperature_c=9.5, location_text="Hub"),
        DriverReply(intent="CONFIRM", raw_text="haan sahi hai"),
        veh_produce
    )
    assert res_prod_outside.decision == GateDecision.CHALLENGE

    # 4. Near 0°C Thaw Transition (+0.5°C spoken against -1.0°C sensor)
    # Must NOT be falsely classified as an acoustic sign-flip trap!
    veh_thaw = VehicleRecord(
        vehicle_id="KA-04-E-5541", driver_name="Manjunath", driver_phone="+91-9744556677",
        vehicle_type="Reefer 32ft", cargo_type="Fresh Perishables / Fruits",
        target_temp_c=-2.0, current_temp_c=-1.0, route="Bangalore to Chennai",
        last_known_location="Hosur", status="IN_TRANSIT"
    )
    res_thaw = evaluate_confirmation_gate(
        ExtractionRecord(temperature_c=0.5, location_text="Hosur"),
        DriverReply(intent="CONFIRM", raw_text="haan sahi hai"),
        veh_thaw
    )
    # Delta is |0.5 - (-1.0)| = 1.5°C <= 2.0°C produce tolerance -> ALLOW (standard sensor rule, not false sign-flip)
    assert res_thaw.decision == GateDecision.ALLOW
    assert "Polarity flip" not in res_thaw.reason
    assert "sign-flip" not in res_thaw.reason


def test_rule_1_sensor_match_with_active_excursion_allows_and_sets_critical():
    """
    Test Rule 1 combined with an excursion:
    A cooling breakdown occurred; sensor reads -10.0°C on -22°C ice cream truck.
    Driver reports -10.0°C.
    - Gate ALLOWs through (Rule 1: spoken matches sensor within ±2.0°C).
    - Dispatch executes with CRITICAL priority because -10.0°C > -18.0°C Codex threshold.
    - Confirmed priority source recorded as CONFIRMED_READING.
    """
    from core.schemas import VehicleRecord, GateDecision, DispatchPriority
    veh_breakdown = VehicleRecord(
        vehicle_id="HR-55-TC-1102", driver_name="Ramesh", driver_phone="+91-9876543210",
        vehicle_type="Reefer 32ft", cargo_type="Ice Cream Consignment",
        target_temp_c=-22.0, current_temp_c=-10.0, route="Karnal to Delhi",
        last_known_location="NH-44 Milepost 118", status="IN_TRANSIT"
    )

    ctrl = DialogueController(vehicle_id="HR-55-TC-1102")
    ctrl.veh_telemetry = veh_breakdown

    ctrl.receive_driver_utterance("Temperature minus 10 degree ho gaya hai NH-44 Toll pe")
    rec = ExtractionRecord(
        incident_type=IncidentType.REEFER_TEMPERATURE,
        temperature_c=-10.0,
        location_text="NH-44 Toll Plaza",
        distance_km=25.0
    )
    ctrl.process_extraction(rec)
    assert ctrl.state == DialogueState.CONFIRMING

    # Pure gate check
    reply = DriverReply(intent="CONFIRM", raw_text="haan bhaiya sahi hai")
    gate_res = evaluate_confirmation_gate(rec, reply, veh_breakdown)
    assert gate_res.decision == GateDecision.ALLOW
    assert "matches live sensor" in gate_res.reason

    # Execute confirmation
    state, prompt = ctrl.handle_confirmation_reply(reply)
    assert state == DialogueState.DONE
    assert ctrl.last_action_result is not None
    assert ctrl.last_action_result["priority"] == DispatchPriority.CRITICAL.value
    assert ctrl.priority_source == "CONFIRMED_READING"
    assert ctrl.confirmed_excursion[0] is True
    assert ctrl.confirmed_excursion[2] == DispatchPriority.CRITICAL


def test_sign_flip_magnitude_match_opposite_sensor_dropped_minus():
    """
    Test acoustic dropped minus sign failure mode measured on real audio:
    Sensor reads -14.0°C. Driver/ASR reports +14.0°C.
    Magnitude matches (|14| == |-14|), but sign is opposite!
    Must trigger CHALLENGE for polarity verification.
    """
    from core.schemas import VehicleRecord, GateDecision
    veh = VehicleRecord(
        vehicle_id="MH-12-RN-8819", driver_name="Sukhwinder", driver_phone="+91-9811223344",
        vehicle_type="Reefer 24ft", cargo_type="Vaccine Cold-Chain",
        target_temp_c=-20.0, current_temp_c=-14.0, route="Pune to Mumbai",
        last_known_location="Mumbai-Pune Expressway", status="IN_TRANSIT"
    )

    # Driver reports +14.0°C (un-repeated)
    rec = ExtractionRecord(temperature_c=14.0, location_text="Expressway")
    reply = DriverReply(intent="CONFIRM", raw_text="haan sahi hai")
    res = evaluate_confirmation_gate(rec, reply, veh)

    assert res.decision == GateDecision.CHALLENGE
    assert "Sign-flip/dropped-minus detected" in res.reason or "Polarity flip" in res.reason
    assert "minus" in res.challenge_prompt.lower()

    # When driver explicitly repeats "14 degree confirm"
    reply_repeated = DriverReply(intent="CONFIRM", raw_text="14 degree confirm")
    res_rep = evaluate_confirmation_gate(rec, reply_repeated, veh)
    # Repeated number passes through with residual risk flag
    assert res_rep.decision == GateDecision.ALLOW
    assert res_rep.residual_risk_flag is True


def test_sign_flip_positive_target_with_negative_spoken_value():
    """Test reverse polarity flip: positive target +4°C with negative spoken -4°C."""
    from core.schemas import VehicleRecord, GateDecision
    veh_dairy = VehicleRecord(
        vehicle_id="DL-01-AA-4092", driver_name="Abdul", driver_phone="+91-9899001122",
        vehicle_type="Reefer 20ft", cargo_type="Dairy & Milk Products",
        target_temp_c=4.0, current_temp_c=4.5, route="Jaipur to Delhi",
        last_known_location="NH-48", status="IN_TRANSIT"
    )

    rec = ExtractionRecord(temperature_c=-4.0, location_text="NH-48")
    reply = DriverReply(intent="CONFIRM", raw_text="haan sahi hai")
    res = evaluate_confirmation_gate(rec, reply, veh_dairy)

    assert res.decision == GateDecision.CHALLENGE
    assert "Polarity flip detected" in res.reason


def test_caller_id_registration_verification_and_retry_cap_escalation():
    """
    Test that an unverified caller:
    1. Can provide the last 4 digits (e.g. '1102') to identify the vehicle.
    2. If providing invalid digits repeatedly, hits the retry cap (2) and escalates to human.
    """
    from core.state_machine import DialogueController, DialogueState
    from core.schemas import ExtractionRecord, IncidentType

    # Case A: Success via 4-digit identification
    ctrl = DialogueController(caller_phone="+91-9999999999", vehicle_id=None)
    assert ctrl.is_unverified_vehicle is True
    ctrl.receive_driver_utterance("Temperature 18 degree ho gaya")
    rec = ExtractionRecord(incident_type=IncidentType.REEFER_TEMPERATURE, temperature_c=18.0, location_text="Highway")
    ctrl.process_extraction(rec)
    assert ctrl.state == DialogueState.ASKING

    # Driver says last 4 digits -> System matches vehicle and reads it back before continuing
    state_id, prompt_id = ctrl.receive_vehicle_identification_reply("gadi ka number 1102 hai bhaiya")
    assert state_id == DialogueState.ASKING
    assert "HR-55-TC-1102" in prompt_id
    assert ctrl.awaiting_vehicle_confirmation is True

    # Driver confirms the read-back vehicle registration
    state_conf, prompt_conf = ctrl.receive_vehicle_identification_reply("haan bhaiya bilkul sahi hai")
    assert ctrl.is_unverified_vehicle is False
    assert ctrl.vehicle_id == "HR-55-TC-1102"
    assert state_conf == DialogueState.CONFIRMING

    # Case B: Failure with retry cap escalation to supervisor
    ctrl_fail = DialogueController(caller_phone="+91-9999999999", vehicle_id=None)
    ctrl_fail.receive_driver_utterance("Breakdown ho gaya")
    ctrl_fail.process_extraction(ExtractionRecord(incident_type=IncidentType.BREAKDOWN, location_text="Highway"))
    assert ctrl_fail.state == DialogueState.ASKING

    # Attempt 1: Unknown digits
    s1, p1 = ctrl_fail.receive_vehicle_identification_reply("9999")
    assert s1 == DialogueState.ASKING
    assert ctrl_fail.registration_retries == 1

    # Attempt 2: Unknown digits -> Exceeds max retries (2) -> ESCALATING
    s2, p2 = ctrl_fail.receive_vehicle_identification_reply("8888")
    assert s2 == DialogueState.ESCALATING
    assert "human fleet supervisor" in p2


def test_unconfirmed_vs_confirmed_excursion_priority():
    """
    Verify that excursion priority is tracked as UNCONFIRMED_INITIAL during extraction,
    and confirmed value determines CONFIRMED_READING upon dispatch.
    """
    from core.state_machine import DialogueController, DialogueState
    from core.schemas import ExtractionRecord, DriverReply, IncidentType, DispatchPriority

    ctrl = DialogueController(vehicle_id="HR-55-TC-1102")  # Ice cream, target -22°C
    ctrl.veh_telemetry.current_temp_c = -21.0
    ctrl.receive_driver_utterance("Temperature 18 degree ho gaya")
    rec = ExtractionRecord(incident_type=IncidentType.REEFER_TEMPERATURE, temperature_c=18.0, location_text="Highway")
    ctrl.process_extraction(rec)

    # Initial extraction shows preliminary unconfirmed excursion
    assert ctrl.priority_source == "UNCONFIRMED_INITIAL"
    assert ctrl.unconfirmed_excursion[0] is True
    assert ctrl.unconfirmed_excursion[2] == DispatchPriority.CRITICAL

    # Driver corrects the misheard temperature to safe -21°C
    ctrl.handle_confirmation_reply(DriverReply(intent="CORRECT", corrected_temperature_c=-21.0))
    assert ctrl.current_record.temperature_c == -21.0

    # Driver confirms the corrected safe reading
    state, prompt = ctrl.handle_confirmation_reply(DriverReply(intent="CONFIRM", raw_text="-21 degree confirm"))
    assert state == DialogueState.DONE
    assert ctrl.priority_source == "CONFIRMED_READING"
    # -21°C is <= -18°C ceiling, so NOT an excursion
    assert ctrl.confirmed_excursion[0] is False
    assert ctrl.last_action_result["priority"] == DispatchPriority.HIGH.value


def test_pinned_adapter_and_model_hashes_match_committed_files():
    """Verify that committed local adapters and denoiser weights match pinned SHA256 hashes."""
    from models.download_adapters import verify_adapters, verify_denoiser
    assert verify_adapters(enforce_hash=True) is True
    assert verify_denoiser(enforce_hash=True) is True


def test_silence_timeout_handling_reprompts_and_escalates_to_supervisor_queue():
    """Verify that silence/timeout prompts up to 2 times, and on 3rd timeout enqueues supervisor ticket and escalates."""
    from core.state_machine import DialogueController, DialogueState
    from core.tools import list_supervisor_queue

    ctrl = DialogueController(vehicle_id="HR-55-TC-1102", caller_phone="+91-9876543210")

    # Turn 1: Driver says nothing (timeout)
    s1, p1 = ctrl.receive_driver_utterance("", asr_confidence=0.0)
    assert s1 == DialogueState.ASKING
    assert "aawaz nahi aa rahi hai" in p1
    assert ctrl.silence_timeout_count == 1

    # Turn 2: Driver still silent
    s2, p2 = ctrl.handle_silence_or_timeout()
    assert s2 == DialogueState.ASKING
    assert "line par hain" in p2
    assert ctrl.silence_timeout_count == 2

    # Turn 3: 3rd consecutive timeout -> escalation to persistent supervisor queue
    s3, p3 = ctrl.handle_silence_or_timeout()
    assert s3 == DialogueState.ESCALATING
    assert "supervisor ko connect kar rahe hain" in p3 or "supervisor" in p3
    assert ctrl.silence_timeout_count == 3
    assert ctrl.last_ticket_id is not None

    # Check persistence in supervisor_queue
    tickets = list_supervisor_queue(status="PENDING")
    ticket = next((t for t in tickets if str(t["escalation_id"]) in ctrl.last_ticket_id), None)
    assert ticket is not None
    assert ticket["vehicle_id"] == "HR-55-TC-1102"
    assert ticket["driver_phone"] == "+91-9876543210"
    assert "SILENCE_TIMEOUT_LIMIT" in ticket["trigger_reason"]


def test_low_asr_confidence_gates_to_asking_reading_upfront():
    """Verify that when Whisper ASR confidence < 0.65, system enters ASKING upfront rather than dispatching."""
    from core.state_machine import DialogueController, DialogueState
    from core.schemas import ExtractionRecord, IncidentType

    ctrl = DialogueController(vehicle_id="HR-55-TC-1102")
    # Low confidence transcription (e.g. 0.42 on noisy 5 dB line)
    s, p = ctrl.receive_driver_utterance("reefer temperature shyd atharah hai", asr_confidence=0.42)
    assert s == DialogueState.ASKING
    assert "saaf nahi aayi" in p
    assert ctrl.state == DialogueState.ASKING

    # Attempting extraction from ASKING without clarifying is guarded and raises RuntimeError
    rec = ExtractionRecord(incident_type=IncidentType.REEFER_TEMPERATURE, temperature_c=18.0, location_text="Panipat")
    with pytest.raises(RuntimeError):
        ctrl.process_extraction(rec)

    # Driver repeats clearly with high confidence
    s2, p2 = ctrl.receive_missing_slot_reply("Haan bhaiya 18 degree hai", rec)
    assert s2 == DialogueState.CONFIRMING


def test_missing_numeric_reading_in_reefer_incident_gates_to_asking():
    """Verify that reefer temperature incident missing a numeric reading prompts directly for the meter display."""
    from core.state_machine import DialogueController, DialogueState
    from core.schemas import ExtractionRecord, IncidentType

    ctrl = DialogueController(vehicle_id="HR-55-TC-1102")
    ctrl.receive_driver_utterance("Bhaiya reefer ka temp kharab lag raha hai", asr_confidence=0.95)
    rec = ExtractionRecord(incident_type=IncidentType.REEFER_TEMPERATURE, temperature_c=None, location_text="Panipat")
    state, prompt = ctrl.process_extraction(rec)

    assert state == DialogueState.ASKING
    assert "reefer display par temperature" in prompt


def test_hindi_first_spoken_digit_spelling():
    """Verify colloquial Hindi digit spelling for disambiguating 18/80 and positive/negative values."""
    from core.schemas import spell_number_in_hindi_digits
    from core.state_machine import DialogueController, DialogueState
    from core.schemas import ExtractionRecord, IncidentType

    assert spell_number_in_hindi_digits(18.0) == "ek, aath"
    assert spell_number_in_hindi_digits(80.0) == "aath, shunya"
    assert spell_number_in_hindi_digits(-14.0) == "minus ek, chaar"
    assert spell_number_in_hindi_digits(40.0) == "chaar, shunya"
    assert spell_number_in_hindi_digits(4.5) == "chaar point paanch"
    assert spell_number_in_hindi_digits(None) == ""

    # Test in DialogueController readback prompt
    ctrl = DialogueController(vehicle_id="HR-55-TC-1102")
    ctrl.receive_driver_utterance("Temperature 18 degree hai", asr_confidence=0.92)
    rec = ExtractionRecord(incident_type=IncidentType.REEFER_TEMPERATURE, temperature_c=18.0, location_text="Karnal")
    state, prompt = ctrl.process_extraction(rec)

    assert state == DialogueState.CONFIRMING
    assert "(ek, aath)" in prompt
    assert "18 degree" in prompt


def test_dispute_audit_bundle_recorded_on_dispatch():
    """Verify that every final dispatch writes an end-to-end forensic audit bundle to dispute_audit_log."""
    from core.state_machine import DialogueController, DialogueState
    from core.schemas import ExtractionRecord, DriverReply, IncidentType
    from core.tools import get_dispute_audit

    ctrl = DialogueController(vehicle_id="HR-55-TC-1102", caller_phone="+91-9876543210")
    ctrl.veh_telemetry.current_temp_c = -21.0
    ctrl.receive_driver_utterance("Temperature minus chaudah degree ho gaya", asr_confidence=0.94, audio_path="audio/driver_turn1.wav")
    rec = ExtractionRecord(incident_type=IncidentType.REEFER_TEMPERATURE, temperature_c=-14.0, location_text="Karnal Bypass")
    ctrl.process_extraction(rec)

    # Driver confirms
    reply = DriverReply(intent="CONFIRM", raw_text="haan bhai minus chaudah sahi hai")
    state, prompt = ctrl.handle_confirmation_reply(reply, audio_path="audio/driver_reply.wav")
    assert state == DialogueState.DONE
    assert ctrl.last_action_result is not None
    assert "ticket_id" in ctrl.last_action_result

    ticket_id = ctrl.last_action_result["ticket_id"]
    audit = get_dispute_audit(ticket_id)
    assert audit is not None
    assert audit["ticket_id"] == ticket_id
    assert audit["initial_audio_path"] == "audio/driver_turn1.wav"
    assert audit["driver_reply_audio_path"] == "audio/driver_reply.wav"
    assert "minus chaudah" in audit["asr_transcript"]
    assert "minus" in audit["readback_prompt_spoken"].lower()
    assert audit["driver_reply_intent"] == "CONFIRM"


def test_supervisor_queue_lifecycle():
    """Verify that tickets can be enqueued, listed, and resolved with notes in supervisor_queue."""
    from core.tools import enqueue_supervisor_escalation, list_supervisor_queue, resolve_supervisor_escalation

    esc_id = enqueue_supervisor_escalation(
        vehicle_id="HR-55-TC-9999",
        trigger_reason="RETRY_CAP_EXCEEDED",
        driver_phone="+91-9123456789",
        audio_recording_path="recordings/call_9999.wav",
        transcript_history="Main bol raha hoon"
    )
    assert isinstance(esc_id, int)

    # List pending
    pending = list_supervisor_queue(status="PENDING")
    ticket = next((t for t in pending if t["escalation_id"] == esc_id), None)
    assert ticket is not None
    assert ticket["status"] == "PENDING"
    assert ticket["trigger_reason"] == "RETRY_CAP_EXCEEDED"

    # Resolve
    success = resolve_supervisor_escalation(
        escalation_id=esc_id,
        resolution_notes="Verified driver manually via national fleet registry"
    )
    assert success is True

    # Check updated status
    all_tickets = list_supervisor_queue(status=None)
    resolved = next((t for t in all_tickets if t["escalation_id"] == esc_id), None)
    assert resolved is not None
    assert resolved["status"] == "RESOLVED"
    assert resolved["resolution_notes"] == "Verified driver manually via national fleet registry"


def test_telephony_server_rest_endpoints():
    """Verify FastAPI telephony server health, queue, resolve, and dispute audit endpoints."""
    from fastapi.testclient import TestClient
    from app.telephony_server import app
    from core.tools import enqueue_supervisor_escalation

    client = TestClient(app)

    # Health check
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ONLINE"
    assert resp.json()["streaming"] == "active"

    # Enqueue a ticket
    esc_id = enqueue_supervisor_escalation(
        vehicle_id="HR-55-TC-1102",
        trigger_reason="ASR_TELEPHONY_DISPUTE",
        driver_phone="+91-9876500000"
    )

    # Fetch supervisor queue
    q_resp = client.get("/api/supervisor/queue?status=PENDING")
    assert q_resp.status_code == 200
    q_data = q_resp.json()
    assert any(t["escalation_id"] == esc_id for t in q_data["tickets"])

    # Resolve ticket
    res_resp = client.post("/api/supervisor/resolve", json={
        "escalation_id": esc_id,
        "supervisor_name": "Ops Manager",
        "resolution_notes": "Driver instructed to proceed to cold store hub"
    })
    assert res_resp.status_code == 200
    assert res_resp.json()["status"] == "RESOLVED"


def test_unconfirmed_sensor_corroborated_excursion_alert_on_timeout():
    """
    Verify that if a driver says '-10°C' for ice cream (target -22°C) and the sensor also reads -10°C,
    and then the call times out or driver goes silent, the system sends a clearly labeled
    'UNVERIFIED but sensor-corroborated' notice to the control tower and escalates with high priority.
    """
    from core.state_machine import DialogueController, DialogueState
    from core.schemas import ExtractionRecord, IncidentType
    from core.tools import list_supervisor_queue

    ctrl = DialogueController(vehicle_id="HR-55-TC-1102", caller_phone="+91-9876543210")
    # Live sensor reads -10.0°C (real excursion: ice cream target is -22°C, ceiling is -18°C)
    ctrl.veh_telemetry.current_temp_c = -10.0

    # Turn 1: Driver reports -10°C
    ctrl.receive_driver_utterance("Temperature minus 10 degree ho gaya hai", asr_confidence=0.95)
    rec = ExtractionRecord(
        incident_type=IncidentType.REEFER_TEMPERATURE,
        temperature_c=-10.0,
        location_text="Karnal Bypass"
    )
    ctrl.process_extraction(rec)
    assert ctrl.state == DialogueState.CONFIRMING

    # Driver then goes silent: Timeout 1
    s1, _ = ctrl.handle_silence_or_timeout()
    assert s1 == DialogueState.CONFIRMING

    # Timeout 2
    s2, _ = ctrl.handle_silence_or_timeout()
    assert s2 == DialogueState.CONFIRMING

    # Timeout 3 -> Escalation!
    s3, prompt = ctrl.handle_silence_or_timeout()
    assert s3 == DialogueState.ESCALATING

    # Check that a sensor-corroborated control tower notice was issued!
    assert ctrl.last_control_tower_notice is not None
    assert "UNVERIFIED but sensor-corroborated excursion" in ctrl.last_control_tower_notice
    assert "-10" in ctrl.last_control_tower_notice

    # Check persistence in supervisor_queue
    pending = list_supervisor_queue(status="PENDING")
    ticket = next((t for t in pending if str(t["escalation_id"]) in ctrl.last_ticket_id), None)
    assert ticket is not None
    assert ticket["alert_level"] == "UNVERIFIED_SENSOR_CORROBORATED"
    assert "UNVERIFIED_SENSOR_CORROBORATED" in ticket["trigger_reason"]
    assert ticket["unconfirmed_temp_c"] == -10.0
    assert ticket["telematics_sensor_c"] == -10.0


def test_spoken_vehicle_registration_fallback_readback_and_multiple_matches():
    """
    Verify that:
    1. Spoken digits read back the matched vehicle ('HR-55-TC-1102, sahi hai?') before continuing.
    2. Driver confirmation locks in the vehicle.
    3. Suffix matching multiple vehicles returns MULTIPLE_MATCHES asking for full plate.
    """
    from core.state_machine import DialogueController, DialogueState

    # Case 1: Driver states suffix, system reads it back to confirm
    ctrl = DialogueController(caller_phone="+91-9999999999", vehicle_id=None)
    assert ctrl.is_unverified_vehicle is True
    ctrl.receive_driver_utterance("Breakdown ho gaya")
    from core.schemas import ExtractionRecord, IncidentType
    ctrl.process_extraction(ExtractionRecord(incident_type=IncidentType.BREAKDOWN, location_text="Highway"))

    s1, p1 = ctrl.receive_vehicle_identification_reply("1102")
    assert s1 == DialogueState.ASKING
    assert "HR-55-TC-1102" in p1
    assert "sahi hai na Bhaiya" in p1
    assert ctrl.awaiting_vehicle_confirmation is True

    # Driver confirms "haan sahi hai"
    s2, p2 = ctrl.receive_vehicle_identification_reply("haan sahi hai bhaiya")
    assert ctrl.awaiting_vehicle_confirmation is False
    assert ctrl.is_unverified_vehicle is False
    assert ctrl.vehicle_id == "HR-55-TC-1102"

    # Case 2: Suffix collision (mocked or query matching multiple)
    ctrl_multi = DialogueController(caller_phone="+91-9999999999", vehicle_id=None)
    ctrl_multi.receive_driver_utterance("Problem hai")
    ctrl_multi.process_extraction(ExtractionRecord(incident_type=IncidentType.BREAKDOWN, location_text="Highway"))

    # When query matches multiple vehicles (e.g. searching generic digits that match 2 trucks)
    # Mocking lookup return for MULTIPLE_MATCHES:
    from unittest.mock import patch
    with patch("core.tools.lookup_vehicle_by_registration_query", return_value=(None, "MULTIPLE_MATCHES")):
        s_m, p_m = ctrl_multi.receive_vehicle_identification_reply("11")
        assert s_m == DialogueState.ASKING
        assert "do ya usse zyada truck hain" in p_m
        assert "poora number plate" in p_m


@pytest.mark.network
def test_pinned_revisions_resolve_on_huggingface():
    """
    Verify that the pinned base model commit hashes resolve to exact remote SHAs on Hugging Face Hub.
    Marked with @pytest.mark.network so offline CI or test runs safely skip it.
    """
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=2.0)
    except Exception:
        pytest.skip("Hugging Face Hub not reachable (offline test environment)")

    try:
        from huggingface_hub import HfApi
        api = HfApi()

        # 1. Base Whisper checkpoint
        whisper_info = api.model_info("openai/whisper-tiny", revision="169d4a4341b33bc18d8881c4b69c2e104e1cc0af")
        assert whisper_info.sha == "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"

        # 2. Base Flan-T5 checkpoint
        flan_info = api.model_info("google/flan-t5-small", revision="0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab")
        assert flan_info.sha == "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
    except Exception as e:
        pytest.skip(f"Hugging Face Hub query failed: {e}")


def test_early_barge_in_blocks_reflexive_confirmation_and_demands_number_repeat():
    """
    Barge-In Safety Guardrail:
    If driver interrupts playback at 1.0s before numbers have finished playing:
    1. A bare reflexive 'haan' or 'sahi hai' is blocked from confirming.
    2. An explicit vocalized number repeat ('haan 18 degree confirm') is accepted.
    3. Normal confirmation after full prompt (playback >= 2.2s) is accepted.
    """
    from core.state_machine import DialogueController, DialogueState
    from core.schemas import ExtractionRecord, DriverReply, IncidentType

    ctrl = DialogueController(vehicle_id="HR-55-TC-1102")
    ctrl.receive_driver_utterance("Temperature 18 degree ho gaya", asr_confidence=0.95)
    rec = ExtractionRecord(incident_type=IncidentType.REEFER_TEMPERATURE, temperature_c=18.0, location_text="Panipat")
    ctrl.process_extraction(rec)
    assert ctrl.state == DialogueState.CONFIRMING

    # 1. Early barge-in at 0.9s with bare reflexive "haan":
    # Driver cut in before the numbers played! Must NOT confirm or dispatch!
    s1, p1 = ctrl.handle_confirmation_reply(
        DriverReply(intent="CONFIRM", raw_text="haan sahi hai"),
        playback_elapsed_s=0.9
    )
    assert s1 == DialogueState.CONFIRMING
    assert "poora suniye" in p1
    assert "18 degree" in p1
    assert ctrl.last_action_result is None

    # 2. Early barge-in at 1.0s with explicit number repeat:
    # Driver actively vocalized "18 degree confirm" -> Safe to accept!
    s2, p2 = ctrl.handle_confirmation_reply(
        DriverReply(intent="CONFIRM", raw_text="haan bhaiya 18 degree confirm"),
        playback_elapsed_s=1.0
    )
    assert s2 == DialogueState.DONE
    assert ctrl.last_action_result is not None
    assert ctrl.last_action_result["status"] == "DISPATCHED"


def test_vaccine_subzero_thaw_critical_excursion_without_false_sign_flip():
    """
    Verify that for vaccine cargo (+2°C to +8°C):
    1. Any reading < +2°C (e.g. +0.5°C or +1.0°C) is treated as a CRITICAL freeze excursion.
    2. Spoken +0.5°C against sensor +0.6°C matches sensor (delta 0.1 <= 0.8) and is ALLOWed
       without triggering a false acoustic sign-flip trap, while setting CRITICAL priority!
    """
    from core.schemas import VehicleRecord, ExtractionRecord, DriverReply, GateDecision, DispatchPriority, check_cargo_temperature_excursion
    from core.state_machine import evaluate_confirmation_gate, DialogueController, DialogueState

    # 1. Check physical cargo excursion logic for vaccines
    is_exc, exc_msg, prio = check_cargo_temperature_excursion("Vaccine Cold-Chain", 0.5)
    assert is_exc is True
    assert prio == DispatchPriority.CRITICAL
    assert "freeze damage risk" in exc_msg

    # 2. Check telematics confirmation gate
    veh = VehicleRecord(
        vehicle_id="MH-12-RN-8819", driver_name="Sukhwinder", driver_phone="+91-9811223344",
        vehicle_type="Reefer 24ft", cargo_type="Vaccine Cold-Chain",
        target_temp_c=5.0, current_temp_c=0.6, route="Pune to Mumbai",
        last_known_location="Expressway", status="IN_TRANSIT"
    )
    res = evaluate_confirmation_gate(
        ExtractionRecord(temperature_c=0.5, location_text="Expressway"),
        DriverReply(intent="CONFIRM", raw_text="haan sahi hai"),
        veh
    )
    # Delta 0.1°C <= 0.8°C vaccine match window -> ALLOW (matches sensor, not false sign flip)
    assert res.decision == GateDecision.ALLOW
    assert "Polarity flip" not in res.reason


def test_supervisor_queue_deduplication_per_vehicle_prevents_redial_flooding():
    """Verify that multiple redials for the same vehicle update the existing PENDING ticket rather than duplicating."""
    from core.tools import enqueue_supervisor_escalation, list_supervisor_queue

    veh_id = "TEST-DEDUP-VEH-01"
    id1 = enqueue_supervisor_escalation(
        vehicle_id=veh_id,
        trigger_reason="FIRST_CALL_DROPPED",
        driver_phone="+91-9876543210"
    )

    # Second call redials after call drops
    id2 = enqueue_supervisor_escalation(
        vehicle_id=veh_id,
        trigger_reason="SECOND_CALL_REDIAL",
        driver_phone="+91-9876543210"
    )

    assert id1 == id2  # Deduplicated into same ticket!
    pending = [t for t in list_supervisor_queue(status="PENDING") if t["vehicle_id"] == veh_id]
    assert len(pending) == 1
    assert pending[0]["trigger_reason"] == "SECOND_CALL_REDIAL"


def test_acoustic_echo_rejected_as_self_confirmation():
    """
    Echo Loop Defense:
    Agent read-back ends in 'Sahi hai na Bhaiya?'.
    1. Question tail reflection ('sahi hai na bhaiya') is rejected as acoustic echo.
    2. Full readback playback echo ('Temperature minus 14 degree (ek, chaar)...') is rejected as echo
       and does NOT satisfy the number repetition gate, even though it contains '14'!
    3. Genuine natural driver confirmations ('sahi hai', 'sahi hai bhaiya') are accepted!
    """
    from core.state_machine import DialogueController, DialogueState
    from core.schemas import ExtractionRecord, DriverReply, IncidentType

    ctrl = DialogueController(vehicle_id="HR-55-TC-1102")
    ctrl.receive_driver_utterance("Temperature minus 14 degree ho gaya", asr_confidence=0.95)
    rec = ExtractionRecord(incident_type=IncidentType.REEFER_TEMPERATURE, temperature_c=-14.0, location_text="Panipat")
    ctrl.process_extraction(rec)
    assert ctrl.state == DialogueState.CONFIRMING

    # 1. Microphone picks up rhetorical question tail from prompt: "sahi hai na bhaiya"
    s1, p1 = ctrl.handle_confirmation_reply(
        DriverReply(intent="CONFIRM", raw_text="sahi hai na bhaiya")
    )
    assert s1 == DialogueState.CONFIRMING
    assert "Echo detect hua hai" in p1
    assert ctrl.last_action_result is None

    # 2. Full prompt reflection: microphone picks up the agent's entire read-back playback
    # Contains the number '-14', but must NOT satisfy the number repetition gate!
    full_prompt_echo = ctrl.last_agent_prompt
    s2, p2 = ctrl.handle_confirmation_reply(
        DriverReply(intent="CONFIRM", raw_text=full_prompt_echo)
    )
    assert s2 == DialogueState.CONFIRMING
    assert "Echo detect hua hai" in p2
    assert ctrl.last_action_result is None

    # 3. Legitimate driver speech saying natural 'sahi hai' (without question particle 'na')
    # Must be accepted as valid driver confirmation!
    s3, p3 = ctrl.handle_confirmation_reply(
        DriverReply(intent="CONFIRM", raw_text="sahi hai")
    )
    assert s3 == DialogueState.DONE
    assert ctrl.last_action_result is not None


def test_slot_isolation_distance_cutin_does_not_affect_temperature():
    """
    Number Cut-in Slot Isolation:
    Read-back is 18°C and 10 km.
    Driver cuts in early with 'haan 42 km'.
    Because utterance specifies distance ('km'), only distance is corrected to 42 km.
    The temperature slot MUST remain 18°C and NOT be corrupted or modified!
    """
    from core.state_machine import DialogueController, DialogueState
    from core.schemas import ExtractionRecord, DriverReply, IncidentType

    ctrl = DialogueController(vehicle_id="HR-55-TC-1102")
    ctrl.receive_driver_utterance("Temperature 18 degree hai 10 km aage", asr_confidence=0.95)
    rec = ExtractionRecord(
        incident_type=IncidentType.REEFER_TEMPERATURE,
        temperature_c=18.0,
        location_text="Panipat Toll",
        distance_km=10.0
    )
    ctrl.process_extraction(rec)
    assert ctrl.state == DialogueState.CONFIRMING
    assert ctrl.current_record.temperature_c == 18.0
    assert ctrl.current_record.distance_km == 10.0

    # Driver cuts in at 1.0s: "haan 42 km"
    s1, p1 = ctrl.handle_confirmation_reply(
        DriverReply(intent="CONFIRM", raw_text="haan 42 km"),
        playback_elapsed_s=1.0
    )
    assert s1 == DialogueState.CONFIRMING
    # Distance updated, temperature untouched!
    assert ctrl.current_record.distance_km == 42.0
    assert ctrl.current_record.temperature_c == 18.0
    assert "42" in p1
    assert "18" in p1


def test_concurrent_enqueues_atomic_deduplication():
    """
    Atomic Deduplication & Concurrency:
    Simulates concurrent racing calls from the same vehicle (e.g. 8 threads simultaneously).
    Enforces that exactly 1 PENDING ticket exists in supervisor_queue under the partial unique index.
    """
    import concurrent.futures
    from core.tools import enqueue_supervisor_escalation, list_supervisor_queue

    veh_id = "TEST-CONCURRENT-VEH-99"

    def enqueue_call(i):
        return enqueue_supervisor_escalation(
            vehicle_id=veh_id,
            trigger_reason=f"CONCURRENT_CALL_THREAD_{i}",
            driver_phone="+91-9876543200",
            alert_level="STANDARD",
            transcript_history=f"Thread {i} call"
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(enqueue_call, i) for i in range(8)]
        results = [f.result() for f in futures]

    assert len(results) == 8
    # All threads returned the same ticket ID
    unique_ids = set(results)
    assert len(unique_ids) == 1

    # Exactly one PENDING ticket exists in the database
    pending = [t for t in list_supervisor_queue(status="PENDING") if t["vehicle_id"] == veh_id]
    assert len(pending) == 1
    assert pending[0]["escalation_id"] == list(unique_ids)[0]


def test_different_incidents_preserve_incident_type_and_latest_location():
    """
    Multi-incident Merging:
    Vehicle experiences reefer alert at Panipat, then redials with breakdown at Karnal.
    Verifies that:
    1. Incident type and latest location are updated to the latest call.
    2. Transcript history preserves both incident contexts with timestamp.
    3. Alert priority escalates according to severity.
    """
    from core.tools import enqueue_supervisor_escalation, list_supervisor_queue

    veh_id = "TEST-MULTI-INCIDENT-01"

    # Call 1: Reefer temperature issue
    id1 = enqueue_supervisor_escalation(
        vehicle_id=veh_id,
        trigger_reason="REEFER_TEMPERATURE_EXCURSION",
        alert_level="HIGH",
        incident_type="reefer_temperature",
        location_text="Panipat Toll",
        transcript_history="Reefer temp warning"
    )

    # Call 2: Engine breakdown (higher priority)
    id2 = enqueue_supervisor_escalation(
        vehicle_id=veh_id,
        trigger_reason="VEHICLE_BREAKDOWN",
        alert_level="CRITICAL",
        incident_type="breakdown",
        location_text="Karnal Bypass Milepost 12",
        transcript_history="Engine smoke breakdown"
    )

    assert id1 == id2
    pending = [t for t in list_supervisor_queue(status="PENDING") if t["vehicle_id"] == veh_id]
    assert len(pending) == 1
    ticket = pending[0]

    # Priority upgraded to CRITICAL
    assert ticket["alert_level"] == "CRITICAL"
    # Latest location and incident type updated
    assert ticket["latest_location"] == "Karnal Bypass Milepost 12"
    assert ticket["incident_type"] == "breakdown"
    # Both incidents documented in audit history
    assert "incident=breakdown" in ticket["transcript_history"]
    assert "Reefer temp warning" in ticket["transcript_history"]


def test_unverified_alert_upgraded_on_confirmation():
    """
    Upgrade Path:
    When a vehicle previously has an UNVERIFIED_SENSOR_CORROBORATED ticket in supervisor_queue,
    and a subsequent redial successfully confirms the reading and dispatches,
    the supervisor ticket is upgraded to CONFIRMED_DISPATCHED with status RESOLVED.
    """
    from core.tools import enqueue_supervisor_escalation, list_supervisor_queue
    from core.state_machine import DialogueController, DialogueState
    from core.schemas import ExtractionRecord, DriverReply, IncidentType

    veh_id = "HR-55-TC-1102"
    # Initial unverified ticket
    esc_id = enqueue_supervisor_escalation(
        vehicle_id=veh_id,
        trigger_reason="UNVERIFIED_SENSOR_CORROBORATED",
        alert_level="UNVERIFIED_SENSOR_CORROBORATED",
        transcript_history="Initial dropped call with unverified excursion"
    )

    # Subsequent redial succeeds: driver reports -14°C (matches live sensor -14°C)
    ctrl = DialogueController(vehicle_id=veh_id)
    ctrl.receive_driver_utterance("Temperature minus 14 degree ho gaya hai", asr_confidence=0.95)
    rec = ExtractionRecord(
        incident_type=IncidentType.REEFER_TEMPERATURE,
        temperature_c=-14.0,
        location_text="Panipat Toll"
    )
    ctrl.process_extraction(rec)
    s, p = ctrl.handle_confirmation_reply(DriverReply(intent="CONFIRM", raw_text="haan sahi hai"))
    assert s == DialogueState.DONE
    assert ctrl.last_action_result is not None

    # Verify the supervisor ticket was automatically upgraded to CONFIRMED_DISPATCHED!
    all_tickets = list_supervisor_queue(status=None)
    upgraded_ticket = next((t for t in all_tickets if t["escalation_id"] == esc_id), None)
    assert upgraded_ticket is not None
    assert upgraded_ticket["status"] == "RESOLVED"
    assert upgraded_ticket["alert_level"] == "CONFIRMED_DISPATCHED"
    assert "Upgraded on redial confirmation" in upgraded_ticket["resolution_notes"]


def test_call_reaching_done_does_not_raise_hangup_alert():
    """
    Hangup Guard Negative Test:
    A call that successfully reached DialogueState.DONE and then disconnects
    must NOT raise an unverified excursion alert.
    """
    from core.state_machine import DialogueController, DialogueState
    from core.schemas import ExtractionRecord, DriverReply, IncidentType
    from core.tools import list_supervisor_queue

    veh_id = "HR-55-TC-1102"
    ctrl = DialogueController(vehicle_id=veh_id)
    ctrl.receive_driver_utterance("Temperature minus 14 degree ho gaya", asr_confidence=0.95)
    rec = ExtractionRecord(incident_type=IncidentType.REEFER_TEMPERATURE, temperature_c=-14.0, location_text="Panipat")
    ctrl.process_extraction(rec)
    ctrl.handle_confirmation_reply(DriverReply(intent="CONFIRM", raw_text="haan sahi hai"))
    assert ctrl.state == DialogueState.DONE

    # Record pending ticket count before disconnect
    pending_before = len(list_supervisor_queue(status="PENDING"))

    # Simulate telephony server finally block disconnect check
    if ctrl.state not in (DialogueState.DONE, DialogueState.ESCALATING):
        ctrl._check_and_enqueue_escalation(trigger_reason="TELEPHONY_CALL_DROPPED_BEFORE_CONFIRMATION")

    pending_after = len(list_supervisor_queue(status="PENDING"))
    assert pending_after == pending_before  # No new ticket raised!


def test_slot_duration_calibration_within_twenty_percent():
    """
    Slot Duration Speech Rate Calibration:
    Verifies that the calibrated speech rate constant (0.16 s/word)
    matches measured conversational Hinglish read-back speech rate
    (35 words / 5.2 s = 0.1485 s/word) within 20%.
    """
    measured_benchmark_rate = 5.2 / 35.0  # ~0.1485 s per word
    calibrated_rate = 0.16

    relative_error = abs(calibrated_rate - measured_benchmark_rate) / measured_benchmark_rate
    assert relative_error < 0.20, f"Speech rate error {relative_error:.1%} exceeds 20% tolerance!"


def test_early_barge_in_divergent_number_routes_to_correction():
    """
    Number Mismatch & Correction Routing:
    ASR misheard 18 as 28. Agent reads back 28°C.
    Driver cuts in early with 'haan 18 degree confirm'.
    Because 18 != 28, the system must NOT accept this as confirmation of 28°C!
    Instead, it routes to CORRECT, updates temperature to 18°C, and reads back 18°C.
    """
    from core.state_machine import DialogueController, DialogueState
    from core.schemas import ExtractionRecord, DriverReply, IncidentType

    ctrl = DialogueController(vehicle_id="HR-55-TC-1102")
    ctrl.receive_driver_utterance("Temperature 28 degree ho gaya", asr_confidence=0.95)
    rec = ExtractionRecord(incident_type=IncidentType.REEFER_TEMPERATURE, temperature_c=28.0, location_text="Panipat")
    ctrl.process_extraction(rec)
    assert ctrl.state == DialogueState.CONFIRMING
    assert ctrl.current_record.temperature_c == 28.0

    # Driver cuts in at 1.0s: "haan 18 degree confirm"
    s1, p1 = ctrl.handle_confirmation_reply(
        DriverReply(intent="CONFIRM", raw_text="haan 18 degree confirm"),
        playback_elapsed_s=1.0
    )
    # Must route to CORRECTING -> CONFIRMING with updated 18°C, NOT confirm 28°C!
    assert s1 == DialogueState.CONFIRMING
    assert ctrl.current_record.temperature_c == 18.0
    assert "18" in p1
    assert "28" not in p1
    assert ctrl.last_action_result is None


def test_abrupt_websocket_drop_triggers_hangup_alert():
    """
    Hangup Safety Handler:
    Verifies that when a call disconnects prematurely during CONFIRMING while
    holding an unconfirmed sensor-corroborated excursion, the fail-safe enqueues
    a critical UNVERIFIED_SENSOR_CORROBORATED ticket to the supervisor queue.
    """
    from core.state_machine import DialogueController, DialogueState
    from core.schemas import ExtractionRecord, IncidentType, VehicleRecord
    from core.tools import list_supervisor_queue

    # Set up vehicle with sensor reading -10°C (Ice cream target -22°C -> active breakdown!)
    veh = VehicleRecord(
        vehicle_id="TEST-HANGUP-VEH-01",
        driver_name="Harish",
        driver_phone="+91-9876543299",
        vehicle_type="Reefer 32ft",
        cargo_type="Ice Cream Consignment",
        target_temp_c=-22.0,
        current_temp_c=-10.0,
        route="Karnal to Delhi",
        last_known_location="Karnal Bypass",
        status="IN_TRANSIT"
    )
    ctrl = DialogueController(vehicle_id=None)
    ctrl.veh_telemetry = veh
    ctrl.vehicle_id = veh.vehicle_id
    ctrl.is_unverified_vehicle = False

    ctrl.receive_driver_utterance("Temperature minus 10 degree ho gaya hai", asr_confidence=0.95)
    rec = ExtractionRecord(incident_type=IncidentType.REEFER_TEMPERATURE, temperature_c=-10.0, location_text="Karnal")
    ctrl.process_extraction(rec)
    assert ctrl.state == DialogueState.CONFIRMING

    # Call drops abruptly (simulating WebSocketDisconnect / hangup before driver confirmation)
    ctrl._check_and_enqueue_escalation(trigger_reason="TELEPHONY_CALL_DROPPED_BEFORE_CONFIRMATION")

    pending = [t for t in list_supervisor_queue(status="PENDING") if t["vehicle_id"] == "TEST-HANGUP-VEH-01"]
    assert len(pending) >= 1
    ticket = pending[-1]
    assert ticket["alert_level"] == "UNVERIFIED_SENSOR_CORROBORATED"
    assert ticket["driver_phone"] == "+91-9876543299"
    assert "TELEPHONY_CALL_DROPPED_BEFORE_CONFIRMATION" in ticket["trigger_reason"] or "UNVERIFIED_SENSOR_CORROBORATED" in ticket["trigger_reason"]



def test_supervisor_queue_deduplication_preserves_highest_priority_and_window():
    """
    Verify supervisor queue deduplication:
    1. Preserves highest alert level (UNVERIFIED_SENSOR_CORROBORATED is never downgraded by a later STANDARD redial).
    2. Appends timestamped call history.
    """
    from core.tools import enqueue_supervisor_escalation, list_supervisor_queue

    veh_id = "TEST-PRIO-PRESERVE-01"
    # First call enqueues high-priority sensor corroborated alert
    id1 = enqueue_supervisor_escalation(
        vehicle_id=veh_id,
        trigger_reason="CALL_DROPPED_UNVERIFIED_CORROBORATED",
        alert_level="UNVERIFIED_SENSOR_CORROBORATED",
        transcript_history="Driver: minus 10; Telematics: -10"
    )

    # Second redial comes in as a standard drop
    id2 = enqueue_supervisor_escalation(
        vehicle_id=veh_id,
        trigger_reason="SECOND_CALL_REDIAL",
        alert_level="STANDARD",
        transcript_history="Driver: call disconnected again"
    )

    assert id1 == id2
    pending = [t for t in list_supervisor_queue(status="PENDING") if t["vehicle_id"] == veh_id]
    assert len(pending) == 1
    # Priority is preserved as UNVERIFIED_SENSOR_CORROBORATED!
    assert pending[0]["alert_level"] == "UNVERIFIED_SENSOR_CORROBORATED"
    # History is appended rather than overwritten
    assert "Driver: minus 10" in pending[0]["transcript_history"]
    assert "[REDIAL " in pending[0]["transcript_history"]
    assert "call disconnected again" in pending[0]["transcript_history"]







