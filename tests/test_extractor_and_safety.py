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

    # For 80C (outside normal reefer bounds), it must set None so controller asks, never rewrite to 18
    rec_80 = extract_incident_details("Temperature 80 degree dikha raha hai Bilaspur Chowk pe")
    assert rec_80.temperature_c is None or rec_80.temperature_c == 80.0


# -------------------------------------------------------------
# 5. Escalation State Consistency
# -------------------------------------------------------------
def test_escalation_state_consistency():
    """Controller must stay in ESCALATING state when max correction rounds are exceeded."""
    ctrl = DialogueController(vehicle_id="HR-55-TC-1102", max_correction_rounds=2)
    ctrl.receive_driver_utterance("Reefer 18 degree toll plaza")
    ctrl.process_extraction(ExtractionRecord(
        incident_type=IncidentType.REEFER_TEMPERATURE,
        temperature_c=18.0,
        location_text="Toll Plaza",
        distance_km=42.0
    ))

    # Round 1 correction
    state1, _ = ctrl.handle_confirmation_reply(DriverReply(intent="CORRECT", corrected_temperature_c=19.0))
    assert state1 == DialogueState.CONFIRMING

    # Round 2 correction
    state2, _ = ctrl.handle_confirmation_reply(DriverReply(intent="CORRECT", corrected_temperature_c=20.0))
    assert state2 == DialogueState.CONFIRMING

    # Round 3 (Exceeds max 2 rounds) -> Escalate
    state3, prompt = ctrl.handle_confirmation_reply(DriverReply(intent="REJECT", raw_text="galat hai"))
    assert state3 == DialogueState.ESCALATING
    assert ctrl.state == DialogueState.ESCALATING
    assert "supervisor" in prompt.lower()
