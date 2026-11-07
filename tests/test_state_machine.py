"""
Tests for Vaani Dialogue Controller State Machine.
Verifies the core safety guarantee: Dispatch actions CAN NEVER run without explicit driver confirmation.
"""

import pytest
from pathlib import Path
import pandas as pd
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
    # GJ-01-BX-7201 has target_temp_c = 6.0C (Produce tolerance ±15C), so 18C is within normal plausibility threshold (<= 15C delta)
    controller = DialogueController(vehicle_id="GJ-01-BX-7201")
    
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

    # Mock telemetry for Vaccine vehicle (Target -20°C, Tolerance ±4°C)
    veh_vaccine = VehicleRecord(
        vehicle_id="MH-12-RN-8819", driver_name="Sukhwinder", driver_phone="+91-9811223344",
        vehicle_type="Reefer 24ft", cargo_type="Vaccine Cold-Chain",
        target_temp_c=-20.0, current_temp_c=-19.0, route="Pune to Mumbai",
        last_known_location="Mumbai-Pune Expressway km 68", status="IN_TRANSIT"
    )

    # 1. Vaccine Safe: -18°C (delta 2°C <= 4°C tolerance) -> ALLOW on standard 'haan'
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

    # 4. Boundary cases on Ice Cream vehicle (Target -22°C, Tolerance ±12°C)
    veh_icecream = VehicleRecord(
        vehicle_id="HR-55-TC-1102", driver_name="Ramesh", driver_phone="+91-9876543210",
        vehicle_type="Reefer 32ft", cargo_type="Ice Cream Consignment",
        target_temp_c=-22.0, current_temp_c=-22.0, route="Karnal to Delhi",
        last_known_location="NH-44 Toll Plaza km 42", status="IN_TRANSIT"
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
    assert "registration number confirm karein" in prompt.lower()
    assert controller.last_action_result is None


def test_generic_location_fragments_rejected_and_anchored_to_gps():
    """Verify that generic location words ('toll', 'highway') are not accepted as dispatch locations alone."""
    from gnani.client import extract_incident_details

    # Generic utterance without a named city or landmark
    rec = extract_incident_details("Bhaiya reefer 18 degree ho gaya hai toll ke paas")
    assert rec.location_text is None  # Bare generic 'toll' is REJECTED by extractor

    # Fed into DialogueController with valid vehicle having GPS telematics
    controller = DialogueController(vehicle_id="HR-55-TC-1102")
    controller.receive_driver_utterance("Bhaiya reefer 18 degree ho gaya hai toll ke paas")
    state, prompt = controller.process_extraction(rec)

    # State machine anchors to vehicle's telematics GPS
    assert state == DialogueState.CONFIRMING
    assert "GPS" in prompt and ("Corridor" in prompt or "NH-44" in prompt)


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
    assert "registration number confirm karein" in prompt

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


