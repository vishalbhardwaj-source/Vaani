"""
Comprehensive Safety and Extraction Regression Tests for Vaani.
Covers:
1. Negative temperature polarity parsing ("minus 14", "minus 20", "-8")
2. Location extraction and default prevention (None returned, no fabricated milepost)
3. Highway milestone discrimination (NH 44 / km 68 is not temperature)
4. Field-aware driver correction parsing (distance vs temperature vs location)
5. Hindi spoken number words ("atharah", "assi", "chalis")
6. Regression test: Adaptive rules MUST NEVER silently rewrite values without driver confirmation
7. Dialogue Controller escalation consistency
"""

import pytest
from core.schemas import DialogueState, ExtractionRecord, IncidentType, DriverReply
from core.state_machine import DialogueController
from gnani.client import extract_incident_details, classify_driver_reply, parse_hindi_number
from core.tools import init_db


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


# -------------------------------------------------------------
# 1. Negative Numbers and Polarity Tests
# -------------------------------------------------------------
def test_negative_temperature_extraction():
    """Verify negative polarity is correctly preserved and not dropped."""
    r1 = extract_incident_details("Temperature minus 14 degree chal raha hai Murthal Dhaba se 10 km pehle")
    assert r1.temperature_c == -14.0
    assert r1.location_text == "Murthal Dhaba"
    assert r1.distance_km == 10.0

    r2 = extract_incident_details("Cabin me beep ho rahi hai minus 20 degree ho gaya Kundli Border se 15 km")
    assert r2.temperature_c == -20.0
    assert r2.location_text == "Kundli Border"
    assert r2.distance_km == 15.0

    r3 = extract_incident_details("Emergency hai ice cream melt ho rahi hai temp minus 8 degree ho gaya Jaipur Highway")
    assert r3.temperature_c == -8.0
    assert r3.location_text == "Jaipur Highway"


# -------------------------------------------------------------
# 2. Defaults Must Not Invent Facts
# -------------------------------------------------------------
def test_missing_location_returns_none_never_fabricated():
    """If no landmark matches, location_text must be None, NOT 'NH Highway Milepost'."""
    r = extract_incident_details("Bhaiya chiller ka temperature 18 degree ho gaya hai jaldi batao")
    assert r.temperature_c == 18.0
    assert r.location_text is None
    # Verifies missing fields detection triggers ASKING state
    missing = r.compute_missing_fields()
    assert "location_text" in missing


def test_unrecognized_incident_returns_unknown():
    """Unrecognized incident must be UNKNOWN, not defaulted to REEFER_TEMPERATURE."""
    r = extract_incident_details("Sir papers check ho rahe hain toll pe")
    assert r.incident_type == IncidentType.UNKNOWN


def test_compound_landmark_precedence():
    """Ensure compound place names match before general substrings."""
    r = extract_incident_details("Breakdown alert gear box jam ho gaya hai Mumbai Pune Expressway km 68")
    assert r.location_text == "Mumbai Pune Expressway"
    assert r.incident_type == IncidentType.BREAKDOWN
    assert r.distance_km == 68.0


def test_highway_route_numbers_not_confused_for_temperature():
    """Road numbers like 'NH 44' or 'km 94' must not be extracted as temperature."""
    r = extract_incident_details("Gaadi heat ho gayi engine band ho gaya NH 44 Milepost 94 pe")
    assert r.incident_type == IncidentType.BREAKDOWN
    assert r.temperature_c is None
    assert "NH 44" in r.location_text or "Milepost" in r.location_text


# -------------------------------------------------------------
# 3. Field-Aware Driver Correction Parsing
# -------------------------------------------------------------
def test_driver_reply_field_discrimination():
    """Verify 'nahi, 42 km hai' updates distance, NOT temperature."""
    reply_dist = classify_driver_reply("Nahi, doori 42 km hai")
    assert reply_dist.intent == "CORRECT"
    assert reply_dist.corrected_distance_km == 42.0
    assert reply_dist.corrected_temperature_c is None

    reply_temp = classify_driver_reply("Nahi, 18 degree hai")
    assert reply_temp.intent == "CORRECT"
    assert reply_temp.corrected_temperature_c == 18.0
    assert reply_temp.corrected_distance_km is None

    reply_loc = classify_driver_reply("Nahi galat hai, Panipat Toll ke paas hoon")
    assert reply_loc.intent == "CORRECT"
    assert "Panipat" in str(reply_loc.corrected_location)


def test_hindi_spoken_number_word_parsing():
    """Spoken Hindi numbers like 'atharah', 'assi', 'chalis' must be resolved."""
    assert parse_hindi_number("atharah") == 18.0
    assert parse_hindi_number("assi") == 80.0
    assert parse_hindi_number("chaudha") == 14.0
    assert parse_hindi_number("chalis") == 40.0
    assert parse_hindi_number("minus chaudha") == -14.0

    # Adversarial spellings and variations
    assert parse_hindi_number("athara") == 18.0
    assert parse_hindi_number("assee") == 80.0
    assert parse_hindi_number("atthais") == 28.0
    assert parse_hindi_number("athais") == 28.0

    # Devanagari script words and numerals
    assert parse_hindi_number("अट्ठारह") == 18.0
    assert parse_hindi_number("अस्सी") == 80.0
    assert parse_hindi_number("अट्ठाइस") == 28.0
    assert parse_hindi_number("चालीस") == 40.0
    assert parse_hindi_number("पैंतीस") == 35.0
    assert parse_hindi_number("१८") == 18.0
    assert parse_hindi_number("८०") == 80.0

    # CRITICAL ADVERSARIAL CASES (Ensures substring misfires do not invent values):
    assert parse_hindi_number("nahi theek nahi hai") is None  # Must NOT match "ek" in "theek"
    assert parse_hindi_number("doori toll se aage") is None    # Must NOT match "do" in "doori"
    assert parse_hindi_number("haan theek hai") is None        # Must NOT match "ek" in "theek"

    # Classify reply regression: "nahi theek nahi hai" is a flat REJECT, NOT a correction to 1.0C
    reply_reject = classify_driver_reply("nahi theek nahi hai")
    assert reply_reject.intent == "REJECT"
    assert reply_reject.corrected_temperature_c is None

    reply = classify_driver_reply("Nahi bhaiya atharah degree hai")
    assert reply.intent == "CORRECT"
    assert reply.corrected_temperature_c == 18.0


# -------------------------------------------------------------
# 4. Regression: Adaptive Rules Never Mutate Values Without Asking
# -------------------------------------------------------------
def test_adaptive_rules_do_not_silently_rewrite_temperatures():
    """
    CRITICAL SAFETY REGRESSION:
    If a real 40C or 80C reefer alarm occurs, the system must NOT rewrite it
    to 14C or 18C behind the scenes. It must keep extracted value and seek confirmation.
    """
    rec_40 = extract_incident_details("Reefer warning aayi hai 40 degree show kar raha hai Bilaspur Chowk pe")
    # Must preserve 40.0, NOT silently rewrite to 14.0
    assert rec_40.temperature_c == 40.0

    # Spoken read-back must speak 40 degree, never 14
    ctrl = DialogueController(vehicle_id="HR-55-TC-1102")
    prompt = ctrl.generate_readback_prompt(rec_40)
    assert "40" in prompt
    assert "14" not in prompt

    # For 80C (outside normal reefer bounds), it must set None, record anomalous_reading, and generate specific prompt
    rec_80 = extract_incident_details("Temperature 80 degree dikha raha hai Bilaspur Chowk pe")
    assert rec_80.temperature_c is None
    assert rec_80.anomalous_reading == 80.0

    ctrl.current_record = rec_80
    clarification_prompt = ctrl._generate_clarification_prompt(["temperature_c"])
    assert "80 degree" in clarification_prompt
    assert "unusual" in clarification_prompt


# -------------------------------------------------------------
# 5. Telematics Plausibility Policy Test & Non-Bypass
# -------------------------------------------------------------
def test_telematics_plausibility_forces_firm_confirmation():
    """
    If reported temperature wildly diverges from vehicle cargo target in fleet.sqlite
    (e.g. reporting 25C when target is -20C for Ice Cream):
    1. A reflex 'haan' is challenged.
    2. A reflex 'haan bilkul sahi hai' does NOT bypass the check.
    3. A bare 'confirm' does NOT bypass the check (bare 'confirm' is as reflexive as 'haan').
    4. The flag does NOT reset after any reflexive reply.
    5. ONLY explicitly repeating the number (e.g. '25 degree') unlocks dispatch.
    """
    # HR-55-TC-1102 has target_temp_c = -20C (Ice Cream)
    ctrl = DialogueController(vehicle_id="HR-55-TC-1102")
    ctrl.receive_driver_utterance("Temperature 25 degree ho gaya hai Toll Plaza pe")
    rec = ExtractionRecord(
        incident_type=IncidentType.REEFER_TEMPERATURE,
        temperature_c=25.0,
        location_text="Toll Plaza",
        distance_km=42.0
    )
    ctrl.process_extraction(rec)
    assert ctrl.state == DialogueState.CONFIRMING
    assert ctrl.requires_forced_confirmation is True

    # Reflex attempt 1: 'haan'
    s1, p1 = ctrl.handle_confirmation_reply(DriverReply(intent="CONFIRM", raw_text="haan"))
    assert s1 == DialogueState.CONFIRMING
    assert ctrl.requires_forced_confirmation is True
    assert "repeat" in p1.lower()

    # Reflex attempt 2: 'bilkul sahi hai'
    s2, _ = ctrl.handle_confirmation_reply(DriverReply(intent="CONFIRM", raw_text="bilkul sahi hai"))
    assert s2 == DialogueState.CONFIRMING
    assert ctrl.requires_forced_confirmation is True

    # Reflex attempt 3: bare 'confirm' (CRITICAL SAFETY REGRESSION)
    s3, _ = ctrl.handle_confirmation_reply(DriverReply(intent="CONFIRM", raw_text="confirm"))
    assert s3 == DialogueState.CONFIRMING
    assert ctrl.requires_forced_confirmation is True

    # Reflex attempt 4: 'haan bilkul confirm'
    s4, _ = ctrl.handle_confirmation_reply(DriverReply(intent="CONFIRM", raw_text="haan bilkul confirm"))
    assert s4 == DialogueState.CONFIRMING
    assert ctrl.requires_forced_confirmation is True

    # Genuine distinct action: driver repeats number
    state_final, _ = ctrl.handle_confirmation_reply(DriverReply(intent="CONFIRM", raw_text="25 degree confirm"))
    assert state_final == DialogueState.DONE
    assert ctrl.requires_forced_confirmation is False
    assert ctrl.last_action_result is not None


# -------------------------------------------------------------
# 6. Correction Path Bounds Enforcement & Transition to ASKING
# -------------------------------------------------------------
def test_correction_path_enforces_bounds():
    """
    If driver speaks an impossible correction (e.g. 99C or -55C):
    1. It must NOT overwrite temperature_c.
    2. Setting temperature_c = None renders record inactionable.
    3. Controller must transition from CONFIRMING -> ASKING.
    4. Dispatch is completely blocked while in ASKING.
    5. Valid clarification reply restores actionability.
    """
    ctrl = DialogueController(vehicle_id="DL-01-AA-4092")
    ctrl.receive_driver_utterance("Reefer 18 degree")
    ctrl.process_extraction(ExtractionRecord(
        incident_type=IncidentType.REEFER_TEMPERATURE,
        temperature_c=18.0,
        location_text="Toll Plaza",
        distance_km=42.0
    ))
    assert ctrl.state == DialogueState.CONFIRMING

    # Driver says an impossible correction: "Nahi 99 degree"
    state_corr, prompt_asking = ctrl.handle_confirmation_reply(
        DriverReply(intent="CORRECT", corrected_temperature_c=99.0, raw_text="nahi 99 degree")
    )
    # Must transition to ASKING because record is missing valid temperature
    assert state_corr == DialogueState.ASKING
    assert ctrl.current_record.temperature_c is None
    assert ctrl.current_record.anomalous_reading == 99.0
    assert "temperature" in prompt_asking.lower()

    # Safety Guarantee: Confirmation reply is impossible from ASKING
    import pytest
    with pytest.raises(RuntimeError):
        ctrl.handle_confirmation_reply(DriverReply(intent="CONFIRM", raw_text="haan"))

    # Safety Guarantee: Direct dispatch is blocked from ASKING
    with pytest.raises(PermissionError):
        ctrl._execute_dispatch_action()

    # Driver supplies valid clarification: -18C
    clarified_record = ctrl.current_record.model_copy()
    clarified_record.temperature_c = -18.0
    state_after_clarify, readback = ctrl.receive_missing_slot_reply("-18 degree", clarified_record)
    assert state_after_clarify == DialogueState.CONFIRMING
    assert "-18" in readback

    # Driver confirms valid readback -> reaches DONE
    state_done, _ = ctrl.handle_confirmation_reply(DriverReply(intent="CONFIRM", raw_text="haan sahi hai"))
    assert state_done == DialogueState.DONE
    assert ctrl.last_action_result["reported_temp_c"] == -18.0


# -------------------------------------------------------------
# 7. Novel Landmark Extraction (Unseen Test Sentences)
# -------------------------------------------------------------
def test_novel_landmark_extraction():
    """Test extractor on novel road phrases and places not in the training corpus."""
    r1 = extract_incident_details("Sir engine band ho gaya hai Karnal bypass pe")
    assert r1.location_text == "Bypass" or "Karnal" in str(r1.location_text)
    assert r1.incident_type == IncidentType.BREAKDOWN

    r2 = extract_incident_details("Bhaiya temperature 15 degree Mathura Highway pe")
    assert r2.temperature_c == 15.0
    assert r2.location_text == "Highway" or "Mathura" in str(r2.location_text)


# -------------------------------------------------------------
# 6. Escalation Safety & Terminal Handover
# -------------------------------------------------------------
def test_escalation_safety_and_supervisor_handover():
    """Controller in ESCALATING state must block dispatch and only terminate via supervisor handover."""
    ctrl = DialogueController(vehicle_id="HR-55-TC-1102", max_correction_rounds=2)
    ctrl.receive_driver_utterance("Reefer 18 degree toll plaza")
    ctrl.process_extraction(ExtractionRecord(
        incident_type=IncidentType.REEFER_TEMPERATURE,
        temperature_c=18.0,
        location_text="Toll Plaza",
        distance_km=42.0
    ))

    # Round 1 correction
    ctrl.handle_confirmation_reply(DriverReply(intent="CORRECT", corrected_temperature_c=19.0))
    # Round 2 correction
    ctrl.handle_confirmation_reply(DriverReply(intent="CORRECT", corrected_temperature_c=20.0))
    # Round 3 -> Escalate
    state_esc, _ = ctrl.handle_confirmation_reply(DriverReply(intent="REJECT", raw_text="galat hai"))
    assert state_esc == DialogueState.ESCALATING

    # CRITICAL: _execute_dispatch_action MUST FAIL in ESCALATING state
    with pytest.raises(PermissionError) as exc_info:
        ctrl._execute_dispatch_action()
    assert "SAFETY GATE VIOLATION" in str(exc_info.value)

    # Clean supervisor resolution
    state_done, msg = ctrl.resolve_escalation_by_supervisor(supervisor_id="SUPERVISOR_PRIYA", action_taken="CALL_DISPATCHED")
    assert state_done == DialogueState.DONE
    assert ctrl.state == DialogueState.DONE
    assert "SUPERVISOR_PRIYA" in msg


# -------------------------------------------------------------
# 9. Substring Bypass Rejection & Strict Numeric Comparison
# -------------------------------------------------------------
def test_substring_bypass_rejected_and_strict_comparison():
    """
    Substrings (e.g. '5' inside '25', or '8' inside '80') MUST NOT bypass
    the telematics forced confirmation gate!
    """
    # Vehicle HR-55-TC-1102 has target_temp -22.0C
    # Case A: Reported temp = 5.0C (|5 - (-22)| = 27C > 25C)
    ctrl = DialogueController(vehicle_id="HR-55-TC-1102")
    ctrl.receive_driver_utterance("Temperature 5 degree hai")
    ctrl.process_extraction(ExtractionRecord(
        incident_type=IncidentType.REEFER_TEMPERATURE,
        temperature_c=5.0,
        location_text="Panipat Toll",
        distance_km=None
    ))
    assert ctrl.requires_forced_confirmation is True

    # Substring trap 1: Driver says 'haan 25 pakka' -> contains '5', but parsed is 25.0 != 5.0
    state1, _ = ctrl.handle_confirmation_reply(DriverReply(intent="CONFIRM", raw_text="haan 25 pakka"))
    assert state1 == DialogueState.CONFIRMING
    assert ctrl.requires_forced_confirmation is True

    # Genuine match: Driver says '5 degree confirm' or 'paanch degree pakka'
    state_ok, _ = ctrl.handle_confirmation_reply(DriverReply(intent="CONFIRM", raw_text="5 degree confirm"))
    assert state_ok == DialogueState.DONE
    assert ctrl.requires_forced_confirmation is False


def test_negative_telematics_plausibility_and_confirmation():
    """
    Cold-chain deep freeze: Vehicle DL-01-AA-4092 has target +4C (Dairy).
    Driver reports -22C (| -22 - 4 | = 26C > 25C).
    1. Driver saying '22 degree confirm' (positive) MUST be rejected.
    2. Driver saying 'minus 22 confirm' or '-22 pakka' MUST unlock dispatch.
    """
    ctrl = DialogueController(vehicle_id="DL-01-AA-4092")
    ctrl.receive_driver_utterance("Sensor -22 degree show kar raha hai")
    ctrl.process_extraction(ExtractionRecord(
        incident_type=IncidentType.REEFER_TEMPERATURE,
        temperature_c=-22.0,
        location_text="Bilaspur Chowk",
        distance_km=None
    ))
    assert ctrl.requires_forced_confirmation is True

    # Wrong polarity: positive 22
    s_wrong, _ = ctrl.handle_confirmation_reply(DriverReply(intent="CONFIRM", raw_text="haan 22 degree confirm"))
    assert s_wrong == DialogueState.CONFIRMING
    assert ctrl.requires_forced_confirmation is True

    # Correct polarity: minus 22
    s_right, _ = ctrl.handle_confirmation_reply(DriverReply(intent="CONFIRM", raw_text="minus 22 degree confirm"))
    assert s_right == DialogueState.DONE
    assert ctrl.requires_forced_confirmation is False
    assert ctrl.last_action_result["reported_temp_c"] == -22.0


def test_flag_lifetime_resets_on_session_reset_and_escalation():
    """requires_forced_confirmation must not leak across sessions or escalations."""
    ctrl = DialogueController(vehicle_id="HR-55-TC-1102")
    ctrl.receive_driver_utterance("Temperature 25 degree")
    ctrl.process_extraction(ExtractionRecord(
        incident_type=IncidentType.REEFER_TEMPERATURE,
        temperature_c=25.0,
        location_text="Toll Plaza",
        distance_km=None
    ))
    assert ctrl.requires_forced_confirmation is True

    # Reset clears flag
    ctrl.reset()
    assert ctrl.requires_forced_confirmation is False

    # Trigger flag again
    ctrl.receive_driver_utterance("Temperature 25 degree")
    ctrl.process_extraction(ExtractionRecord(
        incident_type=IncidentType.REEFER_TEMPERATURE,
        temperature_c=25.0,
        location_text="Toll Plaza",
        distance_km=None
    ))
    assert ctrl.requires_forced_confirmation is True

    # Escalate -> flag cleared
    ctrl.handle_confirmation_reply(DriverReply(intent="REJECT", raw_text="galat hai"))
    ctrl.handle_confirmation_reply(DriverReply(intent="REJECT", raw_text="galat hai"))
    s_esc, _ = ctrl.handle_confirmation_reply(DriverReply(intent="REJECT", raw_text="galat hai"))
    assert s_esc == DialogueState.ESCALATING
    assert ctrl.requires_forced_confirmation is False


def test_unseen_fresh_sentences_slot_accuracy():
    """
    Evaluate slot extraction on 10 completely fresh, unseen sentences
    written independently from the benchmark dataset.
    """
    from gnani.client import extract_incident_details

    fresh_cases = [
        ("Bhaiya reefer 16 degree chala gaya hai Alwar Bypass se 14 km aage", 16.0, "Alwar Bypass", 14.0),
        ("Chiller temperature minus 15 degree ho gaya Manesar Chauraha ke paas", -15.0, "Manesar Chauraha", None),
        ("Engine heat up ho gaya breakdown hai Agra Toll Plaza se 20 km pehle", None, "Agra Toll Plaza", 20.0),
        ("Sir compressor band hai 26 degree display ho raha hai Vapi Checkpost pe", 26.0, "Vapi Checkpost", None),
        ("Reefer warning minus 18 degree Ludhiana GT Road pe", -18.0, "Ludhiana GT Road", None),
        ("Gaadi breakdown ho gayi clutch plate jal gayi Jaipur Bypass km 45", None, "Jaipur Bypass", 45.0),
        ("Bhaiya 22 degree temperature dikha raha hai reefer sensor Karnal Flyover", 22.0, "Karnal Flyover", None),
        ("Sir urgent reefer minus 12 degree pe hai Rohtak Bypass se 8 km", -12.0, "Rohtak Bypass", 8.0),
        ("Cabin me alert hai temperature 32 degree Surat Highway pe", 32.0, "Surat Highway", None),
        ("Reefer temperature 11 degree ho gaya dairy ka Meerut Toll se 16 km", 11.0, "Meerut Toll", 16.0),
    ]

    correct_slots = 0
    total_slots = 0

    for text, expected_temp, expected_loc, expected_dist in fresh_cases:
        rec = extract_incident_details(text)
        
        # Temp check
        total_slots += 1
        if expected_temp is None:
            if rec.temperature_c is None:
                correct_slots += 1
        else:
            if rec.temperature_c is not None and abs(rec.temperature_c - expected_temp) < 0.1:
                correct_slots += 1

        # Loc check
        total_slots += 1
        if expected_loc is None:
            if rec.location_text is None:
                correct_slots += 1
        else:
            if rec.location_text and (expected_loc.lower() in rec.location_text.lower() or rec.location_text.lower() in expected_loc.lower()):
                correct_slots += 1

        # Dist check
        total_slots += 1
        if expected_dist is None:
            if rec.distance_km is None:
                correct_slots += 1
        else:
            if rec.distance_km is not None and abs(rec.distance_km - expected_dist) <= 0.5:
                correct_slots += 1

    accuracy = correct_slots / total_slots
    assert accuracy >= 0.90, f"Fresh sentences slot accuracy {accuracy:.2%} below 90%"


# -------------------------------------------------------------
# 11. Hyphen and Substring Parsing Safety Regressions
# -------------------------------------------------------------
def test_hyphenated_identifiers_do_not_negate_numbers():
    """
    Hyphens in alphanumeric codes (NH-48, km-42, 18-degree) must NOT negate numbers.
    """
    assert parse_hindi_number("NH-48") == 48.0
    assert parse_hindi_number("km-42") == 42.0
    assert parse_hindi_number("18-degree") == 18.0

    # Ensure NH-44 does not corrupt temperature extraction into -44C
    rec = extract_incident_details("NH-44 toll plaza ke paas reefer temp 18 degree hai")
    assert rec.temperature_c == 18.0
    assert rec.anomalous_reading is None

    # Ensure km-42 extracts as positive distance 42 km
    rec_dist = extract_incident_details("Breakdown at km-42 on highway")
    assert rec_dist.distance_km == 42.0


def test_negative_polarity_and_word_boundaries():
    """
    Words like 'barish', 'marine', 'mandir' contain 'rin' or 'in' but must never trigger negation.
    Only strict word-boundary 'minus', 'negative', 'rin', 'rinaatmak', or unary '-' negate.
    """
    assert parse_hindi_number("barish me 18 degree") == 18.0
    assert parse_hindi_number("marine 18 degree") == 18.0
    assert parse_hindi_number("mandir ke paas 18 degree") == 18.0

    # Genuine negative inputs
    assert parse_hindi_number("-22") == -22.0
    assert parse_hindi_number("- 22") == -22.0
    assert parse_hindi_number("minus baais") == -22.0
    assert parse_hindi_number("माइनस बाईस") == -22.0

    # Devanagari 35 should not trigger 30
    assert parse_hindi_number("पैंतीस (35)") == 35.0


def test_two_number_correction_phrases():
    """
    In spoken corrections ('nahi X nahi Y degree'), the final target value must be extracted.
    Adversarial rejections ('nahi theek nahi hai', 'doori toll se aage') must extract no numbers.
    """
    r1 = classify_driver_reply("nahi minus 22 nahi 18 degree")
    assert r1.intent == "CORRECT"
    assert r1.corrected_temperature_c == 18.0

    r2 = classify_driver_reply("nahi 28 nahi 18 degree")
    assert r2.intent == "CORRECT"
    assert r2.corrected_temperature_c == 18.0

    r3 = classify_driver_reply("nahi 18 nahi minus 22 degree")
    assert r3.intent == "CORRECT"
    assert r3.corrected_temperature_c == -22.0

    # Adversarial rejections without numbers
    r_theek = classify_driver_reply("nahi theek nahi hai")
    assert r_theek.intent == "REJECT"
    assert r_theek.corrected_temperature_c is None

    r_doori = classify_driver_reply("doori toll se aage")
    assert r_doori.corrected_distance_km is None
    assert r_doori.intent in ("REJECT", "UNCLEAR")


def test_gate_expected_temperature_five_rejects_and_accepts():
    """
    Gate with expected temperature 5:
    Rejects 'haan 25 pakka' and '25', accepts '5 degree confirm'.
    """
    # Vehicle HR-55-TC-1102 has target_temp_c = -22.0C.
    # Expected temperature 5.0C diverges by 27.0C (> 25C), activating forced confirmation.
    ctrl = DialogueController(vehicle_id="HR-55-TC-1102")
    ctrl.receive_driver_utterance("Temperature 5 degree hai Toll Plaza pe")
    ctrl.process_extraction(ExtractionRecord(
        incident_type=IncidentType.REEFER_TEMPERATURE,
        temperature_c=5.0,
        location_text="Toll Plaza",
        distance_km=10.0
    ))
    assert ctrl.state == DialogueState.CONFIRMING
    assert ctrl.requires_forced_confirmation is True

    # 1. Driver says "haan 25 pakka" -> must be rejected because 25 != 5
    reply_25 = classify_driver_reply("haan 25 pakka")
    s_rej, p_rej = ctrl.handle_confirmation_reply(reply_25)
    assert s_rej == DialogueState.CONFIRMING
    assert ctrl.requires_forced_confirmation is True
    assert "5 degree confirm" in p_rej

    # 2. Driver says "5 degree confirm" -> accepted
    reply_5 = classify_driver_reply("5 degree confirm")
    s_acc, _ = ctrl.handle_confirmation_reply(reply_5)
    assert s_acc == DialogueState.DONE
    assert ctrl.requires_forced_confirmation is False
