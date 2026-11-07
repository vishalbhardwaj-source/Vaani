"""
Vaani: Core Schemas
Pydantic schemas for structured extraction, vehicle telemetry, and dialogue control.
Ensures strict validation before any action tool can be executed.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
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
    CargoClass.PHARMA_VACCINE: 4.0,       # Strict cold-chain (+2°C to +8°C / -20°C): >4°C divergence is dangerous
    CargoClass.DAIRY_CHILLER: 8.0,        # Chilled dairy (+4°C): >8°C divergence requires explicit confirmation
    CargoClass.DEEP_FREEZE: 12.0,         # Sub-zero frozen goods (-18°C to -22°C): >12°C divergence flagged
    CargoClass.PERISHABLE_PRODUCE: 15.0,  # Fresh fruits/vegetables (+6°C to +12°C): >15°C divergence flagged
    CargoClass.AMBIENT_GENERAL: 25.0,     # General ambient cargo: wide tolerance band
}


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
    """Simulated vehicle metadata from fleet database."""
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
    created_at: Optional[str] = None
    status: str = "DISPATCHED"


class StateTransitionLog(BaseModel):
    """Audit log entry for state transitions."""
    timestamp: str
    from_state: DialogueState
    to_state: DialogueState
    trigger: str
    payload: Dict[str, Any] = Field(default_factory=dict)
