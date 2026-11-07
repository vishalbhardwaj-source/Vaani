"""
Vaani: Core Schemas
Pydantic schemas for structured extraction, vehicle telemetry, and dialogue control.
Ensures strict validation before any action tool can be executed.
"""

from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field, field_validator, ConfigDict


class IncidentType(str, Enum):
    REEFER_TEMPERATURE = "reefer_temperature"
    BREAKDOWN = "breakdown"
    DELAY = "delay"
    ACCIDENT = "accident"
    UNKNOWN = "unknown"


class CargoClass(str, Enum):
    PHARMA_VACCINE = "pharma_vaccine"
    DAIRY_CHILLER = "dairy_chiller"
    DEEP_FREEZE = "deep_freeze"
    PERISHABLE_PRODUCE = "perishable_produce"
    AMBIENT_GENERAL = "ambient_general"


CARGO_TOLERANCE_MAP = {
    # Engineering heuristics chosen by the author, informed by cold-chain storage ranges:
    # (Note: Standard documents specify physical storage ceilings, not ASR plausibility bands).
    # 1. Pharma/Vaccines (informed by WHO PQS E003/001 storage range +2°C to +8°C / -20°C): ±4°C heuristic
    CargoClass.PHARMA_VACCINE: 4.0,
    # 2. Dairy & Milk (informed by FSSAI Schedule 4 / HACCP storage norm <= 8°C): ±8°C heuristic
    CargoClass.DAIRY_CHILLER: 8.0,
    # 3. Deep Freeze / Ice Cream (informed by Codex Stan 92-1981 quick frozen food ceiling <= -18°C): ±12°C heuristic
    CargoClass.DEEP_FREEZE: 12.0,
    # 4. Horticultural Produce (informed by APEDA Cold Chain Guidelines 0°C to 15°C): ±15°C heuristic
    CargoClass.PERISHABLE_PRODUCE: 15.0,
    # 5. Configurable default for general ambient freight
    CargoClass.AMBIENT_GENERAL: 25.0,
}

# Empirical tolerance band for matching driver-spoken display temperature with live IoT sensor telemetry.
# Accounts for reefer cab display integer rounding (e.g. -21.6°C displayed as -21°C or -22°C),
# probe calibration offset (±0.5°C), and compartment micro-climatic thermal stratification.
SENSOR_MATCH_TOLERANCE_C: float = 2.0

# Spoken Hindi digit words for noise-resilient digit-by-digit read-backs over 8 kHz telephony
HINDI_DIGIT_MAP: Dict[str, str] = {
    "0": "shunya",
    "1": "ek",
    "2": "do",
    "3": "teen",
    "4": "chaar",
    "5": "paanch",
    "6": "chhah",
    "7": "saat",
    "8": "aath",
    "9": "nau"
}


def spell_number_in_hindi_digits(num: Optional[float]) -> str:
    """
    Spells out individual digits using spoken Hindi colloquial words.
    Crucial over noisy 8 kHz telephony to eliminate confusion between 18 and 80:
    - 18 -> 'ek, aath'
    - 80 -> 'aath, shunya'
    - -14 -> 'minus ek, chaar'
    - 4.5 -> 'chaar point paanch'
    """
    if num is None:
        return ""
    sign_word = "minus " if num < 0 else ""
    abs_val = abs(num)
    if isinstance(num, float) and not num.is_integer():
        # Decimal numbers
        parts = f"{abs_val:g}".split(".")
        int_digits = [HINDI_DIGIT_MAP.get(d, d) for d in parts[0]]
        dec_digits = [HINDI_DIGIT_MAP.get(d, d) for d in parts[1]]
        return f"{sign_word}{', '.join(int_digits)} point {' '.join(dec_digits)}"
    abs_int = abs(int(round(num)))
    digits = [HINDI_DIGIT_MAP.get(d, d) for d in str(abs_int)]
    return f"{sign_word}{', '.join(digits)}"



def get_cargo_tolerance(cargo_type_str: Optional[str]) -> float:
    """Map cargo description to telematics safety tolerance threshold (°C)."""
    if not cargo_type_str:
        return 20.0
    lower = cargo_type_str.lower()
    if "vaccine" in lower or "pharma" in lower or "medicine" in lower:
        return CARGO_TOLERANCE_MAP[CargoClass.PHARMA_VACCINE]
    if "dairy" in lower or "milk" in lower or "chiller" in lower:
        return CARGO_TOLERANCE_MAP[CargoClass.DAIRY_CHILLER]
    if "ice cream" in lower or "frozen" in lower or "meat" in lower or "deep freeze" in lower:
        return CARGO_TOLERANCE_MAP[CargoClass.DEEP_FREEZE]
    if "fruit" in lower or "vegetable" in lower or "perishable" in lower:
        return CARGO_TOLERANCE_MAP[CargoClass.PERISHABLE_PRODUCE]
    return CARGO_TOLERANCE_MAP[CargoClass.AMBIENT_GENERAL]


def check_cargo_temperature_excursion(cargo_type_str: Optional[str], temp_c: Optional[float]) -> Tuple[bool, str, "DispatchPriority"]:
    """
    Evaluates physical cargo thermal excursion independent of speech recognition plausibility.
    Answers: 'Is the cargo at active spoilage/degradation risk?'
    (The ASR Gate answers separately: 'Was this transcribed correctly?').

    Storage thresholds informing excursion detection:
    - Vaccine: safe range +2°C to +8°C (WHO PQS E003/001). Any reading < 0°C (freeze damage) or > 8°C is CRITICAL excursion.
    - Deep Freeze / Ice Cream: critical storage ceiling -18°C (Codex Stan 92-1981, Clause 5.2). Any reading > -18°C is CRITICAL excursion (e.g. -10°C is an excursion!).
    - Dairy & Milk: safe storage <= 8°C (FSSAI Schedule 4, Part II). Reading > 8°C is HIGH excursion; > 12°C is CRITICAL.
    - Fresh Produce: safe storage <= 12°C (APEDA Guidelines). Reading > 15°C is HIGH excursion.
    """
    if temp_c is None:
        return False, "No temperature reported", DispatchPriority.MEDIUM

    cargo = (cargo_type_str or "").lower()
    if "vaccine" in cargo or "pharma" in cargo or "medicine" in cargo:
        if temp_c < 0.0 or temp_c > 8.0:
            return True, f"Vaccine excursion: {temp_c:g}°C outside safe +2°C to +8°C range (WHO PQS E003/001)", DispatchPriority.CRITICAL
    elif "ice cream" in cargo or "frozen" in cargo or "meat" in cargo or "deep freeze" in cargo:
        if temp_c > -18.0:
            return True, f"Deep freeze excursion: {temp_c:g}°C exceeds -18°C storage ceiling (Codex Stan 92-1981)", DispatchPriority.CRITICAL
    elif "dairy" in cargo or "milk" in cargo or "chiller" in cargo:
        if temp_c > 12.0:
            return True, f"Dairy severe excursion: {temp_c:g}°C exceeds 12°C spoil threshold (FSSAI Schedule 4)", DispatchPriority.CRITICAL
        elif temp_c > 8.0:
            return True, f"Dairy moderate excursion: {temp_c:g}°C exceeds 8°C storage ceiling (FSSAI Schedule 4)", DispatchPriority.HIGH
    elif "fruit" in cargo or "vegetable" in cargo or "perishable" in cargo:
        if temp_c > 15.0:
            return True, f"Perishable produce excursion: {temp_c:g}°C exceeds 15°C APEDA ceiling", DispatchPriority.HIGH

    return False, "Temperature within safe storage range", DispatchPriority.HIGH


class GateDecision(str, Enum):
    ALLOW = "ALLOW"          # Permit action tool execution
    CHALLENGE = "CHALLENGE"  # Plausibility anomaly: prompt driver to repeat number
    BLOCK = "BLOCK"          # Safety violation: abort or ask for correction


class GateResult(BaseModel):
    """Result of pure confirmation gate evaluation."""
    decision: GateDecision
    reason: str
    challenge_prompt: Optional[str] = None
    telematics_divergence: Optional[float] = None
    threshold_applied: Optional[float] = None
    residual_risk_flag: bool = False



class DispatchPriority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DialogueState(str, Enum):
    LISTENING = "LISTENING"
    EXTRACTING = "EXTRACTING"
    ASKING = "ASKING"
    CONFIRMING = "CONFIRMING"
    CORRECTING = "CORRECTING"
    ACTING = "ACTING"
    ESCALATING = "ESCALATING"
    DONE = "DONE"


class ExtractionRecord(BaseModel):
    """Structured extraction from driver utterance."""
    model_config = ConfigDict(validate_assignment=True)

    incident_type: IncidentType = Field(
        default=IncidentType.UNKNOWN,
        description="Type of incident reported by the driver"
    )
    vehicle_id: Optional[str] = Field(
        default=None,
        description="Vehicle registration number if mentioned (e.g. HR-55-TC-1102)"
    )
    temperature_c: Optional[float] = Field(
        default=None,
        description="Reefer cabin temperature in Celsius (e.g. 18.0, -14.0)"
    )
    location_text: Optional[str] = Field(
        default=None,
        description="Landmark or highway description (e.g. toll plaza, Bilaspur Chowk)"
    )
    distance_km: Optional[float] = Field(
        default=None,
        description="Distance from landmark in km (e.g. 42.0)"
    )
    consignment_id: Optional[str] = Field(
        default=None,
        description="Cargo or consignment ID if provided"
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="List of required fields missing for action execution"
    )
    raw_transcript: str = Field(
        default="",
        description="Raw ASR transcript from Prisma"
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score of extraction between 0.0 and 1.0"
    )
    anomalous_reading: Optional[float] = Field(
        default=None,
        description="Flagged out-of-bounds reading (e.g. 80C) held for explicit driver clarification"
    )
    acoustic_warning: Optional[str] = Field(
        default=None,
        description="Acoustic confusion warning (e.g. 18 vs 80 / 14 vs 40)"
    )

    @field_validator("temperature_c")
    @classmethod
    def validate_temperature(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            # Physical reefer range check (-40C to +50C)
            if v < -40.0 or v > 50.0:
                raise ValueError(f"Temperature {v}°C out of realistic reefer bounds (-40 to 50)")
        return v

    @field_validator("distance_km")
    @classmethod
    def validate_distance(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            if v < 0 or v > 2000:
                raise ValueError(f"Distance {v} km out of realistic road bounds")
        return v

    def compute_missing_fields(self) -> List[str]:
        missing = []
        if self.incident_type == IncidentType.REEFER_TEMPERATURE:
            if self.temperature_c is None:
                missing.append("temperature_c")
            if not self.location_text:
                missing.append("location_text")
        elif self.incident_type == IncidentType.BREAKDOWN:
            if not self.location_text:
                missing.append("location_text")
        elif self.incident_type == IncidentType.UNKNOWN:
            missing.append("incident_type")
        self.missing_fields = missing
        return missing

    def is_actionable(self) -> bool:
        """Determines if record has all required fields to proceed to confirmation."""
        return len(self.compute_missing_fields()) == 0


class DriverReply(BaseModel):
    """Driver's response to spoken read-back confirmation."""
    intent: str = Field(
        ...,
        description="'CONFIRM' (haan, sahi hai), 'CORRECT' (nahi, 28 nahi 18), 'REJECT' (galat hai), or 'UNCLEAR'"
    )
    corrected_temperature_c: Optional[float] = None
    corrected_location: Optional[str] = None
    corrected_distance_km: Optional[float] = None
    raw_text: str = ""


class VehicleRecord(BaseModel):
    """Simulated vehicle metadata and telematics from fleet database."""
    vehicle_id: str
    driver_name: str
    driver_phone: str
    vehicle_type: str
    cargo_type: str
    target_temp_c: float
    current_temp_c: float
    route: str
    last_known_location: str
    status: str
    gps_lat: Optional[float] = 28.6139
    gps_lng: Optional[float] = 77.2090
    gps_fix_age_minutes: int = 4
    speed_kmh: float = 60.0
    sensor_fault_flag: bool = False


class AssistanceRequest(BaseModel):
    """Simulated dispatch record."""
    ticket_id: Optional[int] = None
    vehicle_id: str
    incident_type: str
    priority: DispatchPriority
    location_reported: str
    distance_km: Optional[float] = None
    reported_temp_c: Optional[float] = None
    notes: Optional[str] = None
    gps_lat: Optional[float] = None
    gps_lng: Optional[float] = None
    gps_fix_timestamp: Optional[str] = None
    supervisor_alert_flag: bool = False
    residual_risk_flag: bool = False
    priority_source: str = "CONFIRMED_READING"  # "CONFIRMED_READING" or "UNCONFIRMED_INITIAL"
    created_at: Optional[str] = None
    status: str = "DISPATCHED"


class StateTransitionLog(BaseModel):
    """Audit log entry for state transitions."""
    timestamp: str
    from_state: DialogueState
    to_state: DialogueState
    trigger: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class SupervisorTicket(BaseModel):
    """Entry in persistent supervisor escalation queue for real human handoffs."""
    escalation_id: Optional[int] = None
    vehicle_id: Optional[str] = None
    driver_phone: Optional[str] = None
    trigger_reason: str
    unconfirmed_temp_c: Optional[float] = None
    telematics_sensor_c: Optional[float] = None
    audio_recording_path: Optional[str] = None
    transcript_history: Optional[str] = None
    status: str = "PENDING"
    created_at: Optional[str] = None
    resolved_at: Optional[str] = None
    resolution_notes: Optional[str] = None


class DisputeAuditEntry(BaseModel):
    """Forensic audit log bundle linking audio, ASR, read-back, reply, and telemetry for disputed dispatches."""
    audit_id: Optional[int] = None
    ticket_id: int
    vehicle_id: str
    initial_audio_path: Optional[str] = None
    asr_transcript: str
    readback_prompt_spoken: str
    driver_reply_audio_path: Optional[str] = None
    driver_reply_intent: str
    driver_reply_text: Optional[str] = None
    gate_decision: str
    residual_risk_flag: bool = False
    priority_source: str = "CONFIRMED_READING"
    dispatched_at: Optional[str] = None

