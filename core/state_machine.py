r"""
Vaani: Dialogue Controller & Finite State Machine
Guarantees the safety gate: Dispatch actions CAN NEVER run without explicit read-back confirmation.

States:
LISTENING -> EXTRACTING -> (ASKING) -> CONFIRMING -> (CORRECTING) -> ACTING -> DONE
                                                    \-> ESCALATING -> DONE
"""

import datetime
import logging
import hashlib
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from core.schemas import (
    DialogueState,
    ExtractionRecord,
    DriverReply,
    IncidentType,
    StateTransitionLog,
    CargoClass,
    CARGO_TOLERANCE_MAP,
    CARGO_SENSOR_MATCH_TOLERANCE_MAP,
    SENSOR_MATCH_TOLERANCE_C,
    get_cargo_sensor_match_tolerance,
    spell_number_in_hindi_digits,
    get_cargo_tolerance,
    GateDecision,
    GateResult,
    DispatchPriority
)
from core.tools import (
    dispatch_assistance,
    update_eta,
    lookup_vehicle,
    lookup_vehicle_by_phone,
    enqueue_supervisor_escalation,
    record_dispute_audit
)

logger = logging.getLogger("vaani.state_machine")


def get_controller_code_hash() -> str:
    """
    Computes a cryptographic SHA256 hash of all benchmark code and data dependencies:
    - core/state_machine.py
    - core/schemas.py
    - core/tools.py
    - core/learning.py
    - gnani/client.py
    - benchmark/run_benchmark.py
    - benchmark/score.py
    - benchmark/utterances.csv
    Normalizes line endings to ensure deterministic cross-platform CI verification.
    """
    repo_root = Path(__file__).resolve().parent.parent
    deps = [
        "benchmark/run_benchmark.py",
        "benchmark/score.py",
        "benchmark/utterances.csv",
        "core/learning.py",
        "core/schemas.py",
        "core/state_machine.py",
        "core/tools.py",
        "gnani/client.py",
    ]

    hasher = hashlib.sha256()
    for rel_path in sorted(deps):
        p = repo_root / rel_path
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                content = f.read().replace("\r\n", "\n")
            hasher.update(rel_path.encode("utf-8"))
            hasher.update(content.encode("utf-8"))

    return hasher.hexdigest()


get_benchmark_dependencies_hash = get_controller_code_hash


def evaluate_confirmation_gate(
    record: ExtractionRecord,
    reply: DriverReply,
    vehicle_telemetry: Optional[Any],
    is_already_challenged: bool = False
) -> GateResult:
    """
    Pure confirmation gate function.
    Evaluates:
    1. Driver affirmative intent (CONFIRM vs CORRECT/REJECT).
    2. Telemetry plausibility: checks both live telematics sensor (current_temp_c)
       and cargo consignment target (target_temp_c) using industry-referenced tolerances.
    3. Challenge requirement for large divergence (> class tolerance).
    4. Explicit number repetition verification to prevent reflexive 'haan' overrides.
    5. Residual risk logging when driver verifies an overriding divergent number.
    """
    intent = reply.intent.upper()
    if intent != "CONFIRM":
        return GateResult(
            decision=GateDecision.BLOCK,
            reason=f"Driver intent '{reply.intent}' is not CONFIRM",
            threshold_applied=None,
            residual_risk_flag=False
        )

    if vehicle_telemetry is None or record.temperature_c is None:
        return GateResult(
            decision=GateDecision.ALLOW,
            reason="No telematics target available; proceeding on driver confirmation alone.",
            threshold_applied=None,
            residual_risk_flag=False
        )

    target_c = vehicle_telemetry.target_temp_c
    sensor_c = getattr(vehicle_telemetry, "current_temp_c", None)
    target_tolerance = get_cargo_tolerance(vehicle_telemetry.cargo_type)
    sensor_match_tolerance = get_cargo_sensor_match_tolerance(vehicle_telemetry.cargo_type)
    fix_age = getattr(vehicle_telemetry, "gps_fix_age_minutes", 0)
    sensor_fault = getattr(vehicle_telemetry, "sensor_fault_flag", False)
    sensor_is_stale = (fix_age > 15)
    sensor_is_valid = (sensor_c is not None and not sensor_fault and not sensor_is_stale)

    from gnani.client import parse_hindi_number
    lower_reply = (reply.raw_text or "").lower()
    parsed_reply_num = parse_hindi_number(lower_reply)
    has_repeated_num = bool(
        parsed_reply_num is not None and
        abs(parsed_reply_num - record.temperature_c) < 0.1
    )

    # 1. Sign-Flip Protection
    # Covers acoustic dropped/added minus sign confusion (e.g. -14°C vs +14°C, -18°C vs +18°C).
    # Near 0°C (-2.0°C to +2.0°C), temperature crossings (e.g. +0.5°C vs -1.0°C) represent physical
    # phase transitions / thawing rather than speech dropped-minus traps.
    is_near_zero_transition = (abs(record.temperature_c) < 2.0 and abs(sensor_c if sensor_c is not None else target_c) < 2.0)

    is_target_polarity_flip = (
        not is_near_zero_transition and
        ((target_c <= -2.0 and record.temperature_c >= 2.0) or (target_c >= 2.0 and record.temperature_c <= -2.0))
    )
    is_sensor_magnitude_sign_flip = (
        not is_near_zero_transition and
        sensor_is_valid and sensor_c != 0.0 and
        (record.temperature_c * sensor_c < 0.0) and
        abs(record.temperature_c) >= 2.0 and abs(sensor_c) >= 2.0 and
        abs(abs(record.temperature_c) - abs(sensor_c)) <= sensor_match_tolerance
    )

    if (is_target_polarity_flip or is_sensor_magnitude_sign_flip) and not has_repeated_num:
        if record.temperature_c > 0.0 and (target_c < 0.0 or (sensor_is_valid and sensor_c < 0.0)):
            ref_val = sensor_c if (is_sensor_magnitude_sign_flip and sensor_is_valid) else target_c
            prompt = f"Negative temperature target/sensor hai ({ref_val:g}°C). Kya aapka matlab minus {abs(record.temperature_c):g} degree tha? Kripya number repeat karein: '{record.temperature_c:g} degree confirm'."
            reason = f"Polarity flip / sign-flip detected: Reported positive {record.temperature_c:g}°C against sub-zero telematics (sensor: {sensor_c if sensor_is_valid else 'N/A'}°C, target: {target_c:g}°C). Explicit sign verification required."
        else:
            ref_val = sensor_c if (is_sensor_magnitude_sign_flip and sensor_is_valid) else target_c
            prompt = f"Positive temperature target/sensor hai ({ref_val:g}°C). Kya aapka matlab {record.temperature_c:g} degree tha? Kripya number repeat karein: '{record.temperature_c:g} degree confirm'."
            reason = f"Polarity flip detected: Reported negative {record.temperature_c:g}°C against positive telematics (sensor: {sensor_c if sensor_is_valid else 'N/A'}°C, target: {target_c:g}°C). Explicit sign verification required."

        ref_div = abs(record.temperature_c - (sensor_c if (is_sensor_magnitude_sign_flip and sensor_is_valid) else target_c))
        applied_tol = sensor_match_tolerance if is_sensor_magnitude_sign_flip else target_tolerance
        return GateResult(
            decision=GateDecision.CHALLENGE,
            reason=reason,
            challenge_prompt=prompt,
            telematics_divergence=ref_div,
            threshold_applied=applied_tol,
            residual_risk_flag=False
        )

    # 2. Live Sensor Decision Rule:
    # - Case 1: Matches Sensor (within tolerance ±2.0°C) -> Strong Evidence ASR is accurate (even if target is far, indicating real breakdown).
    # - Case 2: Matches Target ONLY -> Weak Evidence (sensor indicates divergence, driver spoken target). CHALLENGE.
    # - Case 3: Matches Neither -> Acoustic Trap / Out of bounds. CHALLENGE.
    # - Case 4: Stale or Faulty Sensor -> Fallback to Target-only check.
    if sensor_is_valid:
        delta_sensor = abs(record.temperature_c - sensor_c)
        delta_target = abs(record.temperature_c - target_c)
        matches_sensor = (delta_sensor <= sensor_match_tolerance)
        matches_target = (delta_target <= target_tolerance)

        if matches_sensor:
            # Case 1: Strong evidence that spoken report accurately matches reefer display
            return GateResult(
                decision=GateDecision.ALLOW,
                reason=f"Spoken {record.temperature_c:g}°C matches live sensor {sensor_c:g}°C (delta {delta_sensor:.1f}°C <= {sensor_match_tolerance:g}°C). Strong evidence ASR is accurate.",
                telematics_divergence=delta_sensor,
                threshold_applied=sensor_match_tolerance,
                residual_risk_flag=False
            )
        elif matches_target:
            # Case 2: Weak evidence - matches target, but sensor is divergent!
            if not has_repeated_num:
                return GateResult(
                    decision=GateDecision.CHALLENGE,
                    reason=f"Weak evidence: Spoken {record.temperature_c:g}°C matches target {target_c:g}°C but diverges from live sensor {sensor_c:g}°C (delta {delta_sensor:.1f}°C > {sensor_match_tolerance:g}°C). Explicit repeat required.",
                    challenge_prompt=f"Sensor {sensor_c:g}°C dikha raha hai par report {record.temperature_c:g}°C aayi hai. Kripya display confirm karein: '{record.temperature_c:g} degree confirm'.",
                    telematics_divergence=delta_sensor,
                    threshold_applied=sensor_match_tolerance,
                    residual_risk_flag=False
                )
            else:
                return GateResult(
                    decision=GateDecision.ALLOW,
                    reason=f"Driver verified {record.temperature_c:g}°C override against divergent sensor {sensor_c:g}°C.",
                    telematics_divergence=delta_sensor,
                    threshold_applied=sensor_match_tolerance,
                    residual_risk_flag=True
                )
        else:
            # Case 3: Matches Neither
            if not has_repeated_num:
                comp_label = f"sensor {sensor_c:g}°C / target {target_c:g}°C"
                return GateResult(
                    decision=GateDecision.CHALLENGE,
                    reason=f"Matches neither live sensor ({delta_sensor:.1f}°C > {sensor_match_tolerance:g}°C) nor target ({delta_target:.1f}°C > {target_tolerance:g}°C).",
                    challenge_prompt=f"Telematics {comp_label} se kaafi antar hai. Kripya number repeat karein: '{record.temperature_c:g} degree confirm'.",
                    telematics_divergence=max(delta_sensor, delta_target),
                    threshold_applied=sensor_match_tolerance,
                    residual_risk_flag=False
                )
            else:
                return GateResult(
                    decision=GateDecision.ALLOW,
                    reason=f"Driver explicitly confirmed override of {record.temperature_c:g}°C against telematics ({sensor_c}°C / {target_c}°C).",
                    telematics_divergence=max(delta_sensor, delta_target),
                    threshold_applied=sensor_match_tolerance,
                    residual_risk_flag=True
                )
    else:
        # Case 4: Sensor missing, stale (>15m), or marked faulty -> Fall back to target
        delta_target = abs(record.temperature_c - target_c)
        stale_note = " (sensor stale/faulty)" if sensor_c is not None else ""
        if delta_target <= target_tolerance:
            return GateResult(
                decision=GateDecision.ALLOW,
                reason=f"Reported {record.temperature_c:g}°C within tolerance ({target_tolerance:g}°C) of target {target_c:g}°C{stale_note}.",
                telematics_divergence=delta_target,
                threshold_applied=target_tolerance,
                residual_risk_flag=False
            )
        else:
            if not has_repeated_num:
                return GateResult(
                    decision=GateDecision.CHALLENGE,
                    reason=f"Divergence {delta_target:.1f}°C from target {target_c:g}°C exceeds {target_tolerance:g}°C threshold{stale_note}.",
                    challenge_prompt=f"Telematics target {target_c:g}°C se kaafi antar hai. Kripya number repeat karein: '{record.temperature_c:g} degree confirm'.",
                    telematics_divergence=delta_target,
                    threshold_applied=target_tolerance,
                    residual_risk_flag=False
                )
            else:
                return GateResult(
                    decision=GateDecision.ALLOW,
                    reason=f"Driver verified {record.temperature_c:g}°C override against target {target_c:g}°C{stale_note}.",
                    telematics_divergence=delta_target,
                    threshold_applied=target_tolerance,
                    residual_risk_flag=True
                )


def compute_prompt_segment_end_offset(slots_text: str) -> float:
    """
    Measures the exact audio waveform offset (in seconds) where the critical slots
    (temperature, distance, location) finish playing by inspecting the synthesized audio waveform.
    Uses disk cache to ensure zero latency overhead for repeated prompts.
    Falls back to calibrated conversational speech rate (~0.16 s per word) if TTS is unavailable.
    """
    if not slots_text or not slots_text.strip():
        return 0.0
    try:
        from gnani.client import speak_text
        import io, soundfile as sf
        audio_bytes = speak_text(slots_text)
        data, sr = sf.read(io.BytesIO(audio_bytes))
        measured_s = len(data) / float(sr)
        return round(measured_s, 2)
    except Exception as e:
        logger.debug(f"Audio waveform synthesis measurement fallback: {e}")
        word_count = len(slots_text.split())
        return round(max(1.2, word_count * 0.16), 2)


class DialogueController:
    """
    Finite State Machine governing driver interactions and action gate.
    All transitions are recorded with precise audit trails.
    """
    def __init__(
        self,
        vehicle_id: Optional[str] = "HR-55-TC-1102",
        caller_phone: Optional[str] = None,
        max_correction_rounds: int = 2,
        use_gps_telemetry: bool = True
    ):
        self.state: DialogueState = DialogueState.LISTENING
        self.caller_phone = caller_phone
        self.max_correction_rounds = max_correction_rounds
        self.correction_rounds = 0
        self.use_gps_telemetry = use_gps_telemetry
        self.requires_forced_confirmation = False
        self.last_gate_residual_risk = False
        self.registration_retries = 0
        self.max_registration_retries = 2
        self.silence_timeout_count = 0
        self.max_silence_timeouts = 2
        self.last_audio_path: Optional[str] = None
        self.last_reply_audio_path: Optional[str] = None
        self.last_reply_obj: Optional[DriverReply] = None
        self.asr_confidence: float = 1.0
        self.unconfirmed_excursion: Optional[Tuple[bool, str, DispatchPriority]] = None
        self.confirmed_excursion: Optional[Tuple[bool, str, DispatchPriority]] = None
        self.priority_source: str = "NONE"
        self.pending_vehicle = None
        self.awaiting_vehicle_confirmation = False
        self.last_control_tower_notice: Optional[str] = None
        self.critical_slots_duration_s: float = 2.2
        self.minimum_readback_listen_time_s: float = 2.2

        # Caller-ID lookup if phone number is provided, falling back to vehicle_id
        if caller_phone:
            self.veh_telemetry, phone_reason = lookup_vehicle_by_phone(caller_phone)
            self.vehicle_id = self.veh_telemetry.vehicle_id if self.veh_telemetry else None
            self.unverified_reason = phone_reason if not self.veh_telemetry else None
        elif vehicle_id:
            self.vehicle_id = vehicle_id
            self.veh_telemetry = lookup_vehicle(vehicle_id)
            self.unverified_reason = "VEHICLE_NOT_FOUND" if not self.veh_telemetry else None
        else:
            self.vehicle_id = None
            self.veh_telemetry = None
            self.unverified_reason = "NO_VEHICLE_OR_PHONE_PROVIDED"

        self.is_unverified_vehicle: bool = (self.veh_telemetry is None)

        self.current_record: Optional[ExtractionRecord] = None
        self.last_driver_utterance: str = ""
        self.last_agent_prompt: str = ""
        self.last_action_result: Optional[Dict[str, Any]] = None

        self.history: List[StateTransitionLog] = []
        self._record_transition(
            DialogueState.LISTENING,
            "SESSION_INITIALIZED",
            {"vehicle_id": self.vehicle_id, "caller_phone": self.caller_phone, "verified": not self.is_unverified_vehicle}
        )

    def _record_transition(self, to_state: DialogueState, trigger: str, payload: Dict[str, Any]):
        old_state = self.state
        self.state = to_state
        entry = StateTransitionLog(
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            from_state=old_state,
            to_state=to_state,
            trigger=trigger,
            payload=payload
        )
        self.history.append(entry)

    def _check_and_enqueue_escalation(
        self,
        trigger_reason: str,
        history_summary: Optional[str] = None
    ) -> int:
        """
        Enqueues supervisor ticket with critical corroboration check:
        If dialogue ends in ESCALATING or timeout, but the unconfirmed reading was corroborated
        by the live IoT sensor AND represents a physical cargo excursion, escalates with
        alert_level='UNVERIFIED_SENSOR_CORROBORATED' and registers a prominent Control Tower notice!
        """
        sensor_c = getattr(self.veh_telemetry, "current_temp_c", None) if self.veh_telemetry else None
        cargo_type = self.veh_telemetry.cargo_type if self.veh_telemetry else None
        spoken_temp = self.current_record.temperature_c if self.current_record else None

        alert_level = "STANDARD"
        final_trigger = trigger_reason

        if spoken_temp is not None and sensor_c is not None and cargo_type:
            match_tol = get_cargo_sensor_match_tolerance(cargo_type)
            if abs(spoken_temp - sensor_c) <= match_tol:
                from core.schemas import check_cargo_temperature_excursion
                is_exc, exc_msg, exc_prio = check_cargo_temperature_excursion(cargo_type, spoken_temp)
                if is_exc:
                    alert_level = "UNVERIFIED_SENSOR_CORROBORATED"
                    final_trigger = f"UNVERIFIED_SENSOR_CORROBORATED: Spoken {spoken_temp:g}°C corroborated by live sensor {sensor_c:g}°C ({exc_msg})"
                    self.last_control_tower_notice = (
                        f"[CONTROL TOWER NOTICE - CRITICAL] UNVERIFIED but sensor-corroborated excursion on gaadi {self.vehicle_id or 'UNKNOWN'}! "
                        f"Spoken {spoken_temp:g}°C matches sensor {sensor_c:g}°C (tolerance ±{match_tol:g}°C, {exc_msg})."
                    )

        inc_val = self.current_record.incident_type.value if (self.current_record and hasattr(self.current_record.incident_type, "value")) else None
        loc_val = self.current_record.location_text if self.current_record else None

        ticket_id = enqueue_supervisor_escalation(
            vehicle_id=self.vehicle_id,
            trigger_reason=final_trigger,
            driver_phone=self.caller_phone or (getattr(self.veh_telemetry, "driver_phone", None) if self.veh_telemetry else None),
            unconfirmed_temp_c=spoken_temp,
            telematics_sensor_c=sensor_c,
            audio_recording_path=self.last_audio_path,
            transcript_history=history_summary or "; ".join([f"{h.from_state}->{h.to_state}: {h.trigger}" for h in self.history[-5:]]),
            alert_level=alert_level,
            incident_type=inc_val,
            location_text=loc_val
        )
        self.last_ticket_id = f"TICK-{ticket_id}" if isinstance(ticket_id, int) else str(ticket_id)
        return ticket_id

    def handle_silence_or_timeout(self) -> Tuple[DialogueState, str]:
        """
        Models voice activity timeout: driver stays silent, background road noise dominates,
        or telephony packet loss occurs.
        Enforces a 2-retry schedule before escalating to a human supervisor queue.
        """
        self.silence_timeout_count += 1
        if self.silence_timeout_count > self.max_silence_timeouts:
            self._record_transition(
                DialogueState.ESCALATING,
                trigger="SILENCE_TIMEOUT_EXCEEDED",
                payload={"silence_count": self.silence_timeout_count}
            )
            self._check_and_enqueue_escalation(trigger_reason="SILENCE_TIMEOUT_LIMIT_EXCEEDED")
            prompt = (
                "Bhaiya, aawaz nahi aa rahi hai. Main call turant human fleet supervisor ko "
                "connect kar raha hoon. Kripya line par bane rahein."
            )
            self.last_agent_prompt = prompt
            return DialogueState.ESCALATING, prompt

        if self.state == DialogueState.CONFIRMING and self.current_record:
            temp_desc = f"{self.current_record.temperature_c:g} degree" if self.current_record.temperature_c is not None else "report"
            if self.silence_timeout_count == 1:
                prompt = f"Bhaiya, aapki aawaz nahi aa rahi hai. Kya temperature {temp_desc} confirm hai? Haan ya naa bole."
            else:
                prompt = f"Kya aap line par hain? Kripya confirm karein: kya {temp_desc} sahi hai?"
            self.last_agent_prompt = prompt
            return DialogueState.CONFIRMING, prompt
        else:
            self._record_transition(
                DialogueState.ASKING,
                trigger="SILENCE_TIMEOUT_REPROMPT",
                payload={"silence_count": self.silence_timeout_count}
            )
            if self.silence_timeout_count == 1:
                prompt = "Aapki aawaz nahi aa rahi hai. Kripya batayein kya pareshani hai?"
            else:
                prompt = "Kya aap line par hain? Kripya gaadi ki problem ya temperature batayein."
            self.last_agent_prompt = prompt
            return DialogueState.ASKING, prompt

    def receive_driver_utterance(
        self,
        transcript: str,
        asr_confidence: float = 1.0,
        audio_path: Optional[str] = None
    ) -> Tuple[DialogueState, str]:
        """
        Step 1: Driver speaks in LISTENING state.
        Transitions: LISTENING -> EXTRACTING
        Handles silence/timeout and low-confidence ASR gating.
        """
        if self.state not in (DialogueState.LISTENING, DialogueState.DONE, DialogueState.ASKING):
            raise RuntimeError(f"Cannot receive initial utterance while in state {self.state}")

        self.last_audio_path = audio_path
        self.asr_confidence = asr_confidence

        raw = (transcript or "").strip()
        # Silence / timeout detection
        if not raw or raw in ("[SILENCE]", "[NO_SPEECH]"):
            return self.handle_silence_or_timeout()

        self.silence_timeout_count = 0  # Reset on valid driver speech
        self.last_driver_utterance = raw
        self.correction_rounds = 0
        self.last_action_result = None

        # Low-Confidence / Out-of-Vocabulary Acoustic Safety Check
        if asr_confidence < 0.65:
            self._record_transition(
                DialogueState.ASKING,
                trigger="LOW_ASR_CONFIDENCE",
                payload={"raw_transcript": raw, "confidence": asr_confidence}
            )
            prompt = "Aapki aawaz saaf nahi aayi. Kripya display par dikh raha temperature number saaf batayein (jaise minus 14 ya 18)."
            self.last_agent_prompt = prompt
            return self.state, prompt
        
        self._record_transition(
            DialogueState.EXTRACTING,
            trigger="DRIVER_INITIAL_REPORT",
            payload={"raw_transcript": self.last_driver_utterance, "confidence": asr_confidence}
        )
        return self.state, f"Processing report: '{self.last_driver_utterance}'"

    def process_extraction(self, record: ExtractionRecord) -> Tuple[DialogueState, str]:
        """
        Step 2: Evon extracts structured fields.
        Transitions: EXTRACTING -> CONFIRMING (if actionable) or ASKING (if missing critical slots)
        """
        if self.state != DialogueState.EXTRACTING:
            raise RuntimeError(f"Extraction can only be processed from EXTRACTING state, currently: {self.state}")

        self.current_record = record

        # Validate vehicle ID verification
        if self.is_unverified_vehicle or not self.vehicle_id:
            self._record_transition(
                DialogueState.ASKING,
                trigger="UNVERIFIED_VEHICLE_ID",
                payload={"provided_vehicle_id": self.vehicle_id, "reason": getattr(self, "unverified_reason", None)}
            )
            if getattr(self, "unverified_reason", None) == "MULTIPLE_MATCHES":
                prompt = "Aapka phone number ek se zyada gaadiyon se linked hai. Kripya gaadi ke registration number ke aakhri 4 digit batayein (jaise 1102)."
            else:
                prompt = "Bhaiya gaadi ka registration number ya number plate ke aakhri 4 digit batayein taaki telematics verify ho sake (jaise 1102)."
            self.last_agent_prompt = prompt
            return self.state, prompt

        if not record.vehicle_id:
            record.vehicle_id = self.vehicle_id

        # Telematics location anchor: Only anchor to GPS if explicitly enabled (disabled for live testing)
        if self.use_gps_telemetry and not record.location_text:
            if self.veh_telemetry and getattr(self.veh_telemetry, "last_known_location", None):
                record.location_text = f"GPS: {self.veh_telemetry.last_known_location}"

        # Out-of-Vocabulary / Missing Numeric Reading Gate for Reefer Temperature Incidents:
        if record.incident_type == IncidentType.REEFER_TEMPERATURE and record.temperature_c is None:
            self._record_transition(
                DialogueState.ASKING,
                trigger="NO_NUMERIC_READING_DETECTED",
                payload={"raw_transcript": self.last_driver_utterance}
            )
            prompt = "Bhaiya, reefer display par temperature kitna degree dikh raha hai? Saaf batayein."
            self.last_agent_prompt = prompt
            return self.state, prompt

        # Preliminary physical excursion check on unconfirmed initial reading
        veh = self.veh_telemetry or lookup_vehicle(record.vehicle_id or self.vehicle_id)
        from core.schemas import check_cargo_temperature_excursion
        is_exc, exc_msg, exc_prio = check_cargo_temperature_excursion(
            veh.cargo_type if veh else None,
            record.temperature_c
        )
        self.unconfirmed_excursion = (is_exc, exc_msg, exc_prio)
        self.priority_source = "UNCONFIRMED_INITIAL"

        if not record.is_actionable():
            missing = record.compute_missing_fields()
            self._record_transition(
                DialogueState.ASKING,
                trigger="MISSING_CRITICAL_SLOTS",
                payload={"missing_fields": missing, "record": record.model_dump()}
            )
            prompt = self._generate_clarification_prompt(missing)
            self.last_agent_prompt = prompt
            return self.state, prompt

        # Record is actionable -> proceed to CONFIRMING
        self._record_transition(
            DialogueState.CONFIRMING,
            trigger="RECORD_ACTIONABLE_AWAITING_CONFIRMATION",
            payload={"record": record.model_dump()}
        )
        prompt = self.generate_readback_prompt(self.current_record)
        self.last_agent_prompt = prompt
        return self.state, prompt

    def receive_vehicle_identification_reply(self, reply_text: str) -> Tuple[DialogueState, str]:
        """
        Processes driver utterance providing vehicle registration (last 4 digits or full plate).
        Enforces:
        1. When spoken digits match a single vehicle, reads it back to confirm ("HR-55-TC-1102, sahi hai?")
           before proceeding, preventing incorrect telematics binding.
        2. When multiple vehicles match the suffix (MULTIPLE_MATCHES), prompts for full plate.
        3. Enforces a 2-retry cap before escalating to supervisor.
        """
        clean_text = (reply_text or "").strip().lower()

        # Step 2: Driver answering read-back confirmation of matched vehicle
        if getattr(self, "awaiting_vehicle_confirmation", False) and self.pending_vehicle:
            # Positive confirmation
            if any(w in clean_text for w in ["haan", "sahi", "yes", "theek", "right", "correct", "confirm"]):
                veh = self.pending_vehicle
                self.vehicle_id = veh.vehicle_id
                self.veh_telemetry = veh
                self.is_unverified_vehicle = False
                self.unverified_reason = None
                self.registration_retries = 0
                self.awaiting_vehicle_confirmation = False
                self.pending_vehicle = None
                self._record_transition(
                    DialogueState.EXTRACTING,
                    trigger="VEHICLE_CONFIRMED_BY_DRIVER",
                    payload={"vehicle_id": veh.vehicle_id}
                )
                if self.current_record:
                    self.current_record.vehicle_id = veh.vehicle_id
                    return self.process_extraction(self.current_record)
                else:
                    prompt = f"Gaadi {veh.vehicle_id} verify ho gayi hai. Batayein kya samasya hai?"
                    self.last_agent_prompt = prompt
                    return DialogueState.LISTENING, prompt
            # Negative / rejection
            elif any(w in clean_text for w in ["nahi", "nahin", "galat", "no", "wrong"]):
                self.pending_vehicle = None
                self.awaiting_vehicle_confirmation = False
                self.registration_retries += 1
                if self.registration_retries >= self.max_registration_retries:
                    self._record_transition(
                        DialogueState.ESCALATING,
                        trigger="EXCEEDED_MAX_REGISTRATION_RETRIES",
                        payload={"attempts": self.registration_retries, "last_query": reply_text}
                    )
                    self._check_and_enqueue_escalation(trigger_reason="REGISTRATION_RETRY_LIMIT_EXCEEDED")
                    prompt = (
                        "Bhaiya, do baar gaadi verify nahi ho paayi. Main call turant human fleet supervisor "
                        "ko transfer kar raha hoon. Kripya line par bane rahein."
                    )
                    self.last_agent_prompt = prompt
                    return DialogueState.ESCALATING, prompt
                else:
                    prompt = "Theek hai. Kripya gaadi ke number plate ke aakhri 4 digit dobara saaf batayein (jaise 1102)."
                    self.last_agent_prompt = prompt
                    return DialogueState.ASKING, prompt

        from core.tools import lookup_vehicle_by_registration_query
        veh, match_status = lookup_vehicle_by_registration_query(reply_text)

        if match_status == "MATCH_FOUND" and veh:
            # Spoken digits fallback: read matched vehicle back to confirm
            self.pending_vehicle = veh
            self.awaiting_vehicle_confirmation = True
            prompt = f"Gaadi {veh.vehicle_id} identify hui hai, sahi hai na Bhaiya?"
            self.last_agent_prompt = prompt
            return DialogueState.ASKING, prompt
        elif match_status == "MULTIPLE_MATCHES":
            self.registration_retries += 1
            if self.registration_retries >= self.max_registration_retries:
                self._record_transition(
                    DialogueState.ESCALATING,
                    trigger="EXCEEDED_MAX_REGISTRATION_RETRIES",
                    payload={"attempts": self.registration_retries, "status": match_status}
                )
                self._check_and_enqueue_escalation(trigger_reason="MULTIPLE_MATCHES_RETRY_LIMIT_EXCEEDED")
                prompt = (
                    "Bhaiya, fleet mein is suffix ke ek se zyada truck hain. Main call turant human fleet supervisor "
                    "ko transfer kar raha hoon. Kripya line par bane rahein."
                )
                self.last_agent_prompt = prompt
                return DialogueState.ESCALATING, prompt
            else:
                prompt = "Is number ke do ya usse zyada truck hain. Kripya poora number plate batayein (jaise HR-55-TC-1102)."
                self.last_agent_prompt = prompt
                return DialogueState.ASKING, prompt
        else:  # NOT_FOUND
            self.registration_retries += 1
            if self.registration_retries >= self.max_registration_retries:
                self._record_transition(
                    DialogueState.ESCALATING,
                    trigger="EXCEEDED_MAX_REGISTRATION_RETRIES",
                    payload={"attempts": self.registration_retries, "last_query": reply_text}
                )
                self._check_and_enqueue_escalation(trigger_reason="REGISTRATION_NOT_FOUND_RETRY_LIMIT_EXCEEDED")
                prompt = (
                    "Bhaiya, do baar gaadi verify nahi ho paayi. Main call turant human fleet supervisor "
                    "ko transfer kar raha hoon. Kripya line par bane rahein."
                )
                self.last_agent_prompt = prompt
                return DialogueState.ESCALATING, prompt
            else:
                prompt = "Gaadi number mila nahi. Kripya number plate ke aakhri 4 digit saaf batayein (jaise 1102)."
                self.last_agent_prompt = prompt
                return DialogueState.ASKING, prompt

    def receive_missing_slot_reply(self, answer_transcript: str, updated_record: ExtractionRecord) -> Tuple[DialogueState, str]:
        """
        Handle driver response to clarification question while in ASKING state.
        Transitions: ASKING -> EXTRACTING -> process_extraction()
        """
        if self.state != DialogueState.ASKING:
            raise RuntimeError(f"Cannot process clarification reply from state {self.state}")

        if self.is_unverified_vehicle:
            return self.receive_vehicle_identification_reply(answer_transcript)

        self.last_driver_utterance = answer_transcript
        self._record_transition(
            DialogueState.EXTRACTING,
            trigger="DRIVER_CLARIFICATION_RECEIVED",
            payload={"answer": answer_transcript}
        )
        return self.process_extraction(updated_record)

    def generate_readback_prompt(
        self,
        record: ExtractionRecord,
        force_digit_spelling: bool = False,
        use_hindi_digits: bool = True
    ) -> str:
        """
        Generate spoken Hinglish read-back string for Timbre TTS.
        Normal read-back is concise (~3.5 s). Digit-by-digit spelling is reserved for
        when the telematics gate challenges an anomalous reading or in extreme noise conditions.
        Supports Hindi-first digit spelling ('ek, aath' vs 'aath, shunya') to resolve 18/80 confusions.
        Supports full barge-in / interruptibility during playback.
        Example normal: 'Temperature 18 degree, toll plaza se 42 km aage. Sahi hai na Bhaiya?'
        Example challenged: 'Temperature 80 degree (aath, shunya), toll plaza se 42 (chaar, do km) aage. Dhyan dein: ...'
        """
        # Telematics plausibility policy: Flag deviation beyond per-cargo class tolerance upfront
        telematics_note = None
        if self.veh_telemetry and record.temperature_c is not None:
            tolerance = get_cargo_tolerance(self.veh_telemetry.cargo_type)
            sensor_c = getattr(self.veh_telemetry, "current_temp_c", None)
            target_c = self.veh_telemetry.target_temp_c
            target_diff = abs(record.temperature_c - target_c)
            sensor_diff = abs(record.temperature_c - sensor_c) if sensor_c is not None else target_diff
            temp_diff = max(target_diff, sensor_diff)
            if temp_diff > tolerance:
                self.requires_forced_confirmation = True
                comp = f"Target {int(target_c)} degree tha" if sensor_c is None else f"Sensor {int(sensor_c)} degree aur target {int(target_c)} degree tha"
                telematics_note = f"Dhyan dein: {comp} aur report {int(record.temperature_c)} degree hai"

        use_spelling = force_digit_spelling or getattr(self, "requires_forced_confirmation", False)

        parts = []
        if record.temperature_c is not None:
            temp_num = int(record.temperature_c) if record.temperature_c.is_integer() else record.temperature_c
            sign_word = "minus " if record.temperature_c < 0 else ""
            if use_spelling:
                if use_hindi_digits:
                    hindi_spelled = spell_number_in_hindi_digits(record.temperature_c)
                    parts.append(f"Temperature {sign_word}{abs(temp_num)} degree ({hindi_spelled})")
                else:
                    digits = ", ".join(list(str(abs(int(temp_num)))))
                    parts.append(f"Temperature {sign_word}{temp_num} degree ({sign_word}{digits})")
            else:
                parts.append(f"Temperature {sign_word}{temp_num} degree")

        if record.location_text:
            if record.location_text.startswith("GPS: "):
                if self.use_gps_telemetry:
                    age_str = f" ({self.veh_telemetry.gps_fix_age_minutes} minute purana)" if (self.veh_telemetry and getattr(self.veh_telemetry, "gps_fix_age_minutes", None)) else ""
                    parts.append(f"GPS telemetry{age_str} ke mutabik {record.location_text[5:]}")
                else:
                    parts.append(f"{record.location_text[5:]} ke paas")
            elif record.distance_km is not None:
                dist_num = int(record.distance_km) if record.distance_km.is_integer() else record.distance_km
                if use_spelling:
                    if use_hindi_digits:
                        dist_spelled = spell_number_in_hindi_digits(record.distance_km)
                        parts.append(f"{record.location_text} se {dist_num} ({dist_spelled} km) aage")
                    else:
                        dist_digits = ", ".join(list(str(abs(int(dist_num)))))
                        parts.append(f"{record.location_text} se {dist_num} ({dist_digits}) km aage")
                else:
                    parts.append(f"{record.location_text} se {dist_num} km aage")
            else:
                parts.append(f"{record.location_text} ke paas")

        if record.incident_type == IncidentType.BREAKDOWN:
            parts.insert(0, f"Vehicle {self.vehicle_id} breakdown report")

        if telematics_note:
            parts.append(telematics_note)

        details = ", ".join(parts) if parts else "Diya gaya report"
        # Measured waveform duration for critical slots:
        # Measures the exact audio waveform end offset (seconds) where the critical slots
        # finish playing, using real synthesized waveform lengths.
        # If offline or TTS synthesis is unavailable, falls back to the calibrated rate (0.16 s/word).
        self.critical_slots_end_s = compute_prompt_segment_end_offset(details)
        self.critical_slots_duration_s = self.critical_slots_end_s
        return f"{details}. Sahi hai na Bhaiya?"

    def _is_acoustic_echo(self, raw_reply: str) -> bool:
        """
        Detects acoustic echo loops where the driver's microphone picks up the agent's
        own playback (carrier line echo, speakerphone leakage).
        Prevents full readback loops (e.g. prompt repeating back) or prompt tail loops
        ('sahi hai na bhaiya') from self-confirming.
        Allows genuine natural conversational confirmations like 'sahi hai', 'sahi hai bhaiya',
        'haan', 'theek hai', 'confirm', and explicit repetition responses ('18 degree confirm').
        """
        if not raw_reply or not self.last_agent_prompt:
            return False
        import difflib, re
        clean_reply = re.sub(r"[^\w\s]", "", raw_reply.lower()).strip()
        clean_prompt = re.sub(r"[^\w\s]", "", self.last_agent_prompt.lower()).strip()
        if not clean_reply or not clean_prompt:
            return False

        reply_words = clean_reply.split()

        # 1. Full or substantial playback echo of the agent prompt
        seq_ratio = difflib.SequenceMatcher(None, clean_reply, clean_prompt).ratio()
        if seq_ratio >= 0.65:
            return True

        # Check if reply contains a large block (>= 5 words) that is an exact substring of the prompt
        if len(reply_words) >= 5 and clean_reply in clean_prompt:
            return True

        # 2. Rhetorical question echo loop:
        # Repeating the question particle 'na' with 'sahi' or 'theek' (e.g. 'sahi hai na bhaiya', 'sahi hai na').
        # Genuine driver confirmations say 'sahi hai' or 'sahi hai bhaiya' WITHOUT 'na'!
        if "na" in reply_words and any(w in reply_words for w in ["sahi", "theek"]):
            tail_tokens = {"sahi", "hai", "na", "bhaiya", "kya", "yeh"}
            if set(reply_words).issubset(tail_tokens):
                return True

        return False

    def handle_confirmation_reply(
        self,
        reply: DriverReply,
        reply_audio_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        playback_elapsed_s: Optional[float] = None
    ) -> Tuple[DialogueState, str]:
        """
        Step 3: Evaluate driver reply to read-back using pure confirmation gate.
        Rules:
        - CONFIRM -> gate evaluates plausibility:
            - If ALLOW -> transitions to ACTING (and immediately executes dispatch) -> DONE
            - If CHALLENGE -> prompts for explicit number repetition
        - CORRECT -> transitions to CORRECTING -> CONFIRMING
        - REJECT / UNCLEAR -> treated as not confirmed; retries or escalates
        - If correction_rounds exceeds max_correction_rounds -> ESCALATING to persistent supervisor queue
        - Early Barge-in Guardrail: If driver cut in before numbers finished playing (dynamic critical_slots_duration_s),
          a reflexive 'haan' is rejected unless the driver explicitly repeats the matching number.
        - Number Mismatch Guardrail: If driver cuts in with a number different from readback (e.g. readback 80°C, driver
          says 'haan 18 confirm'), routes to CORRECT path instead of accepting misheard number.
        - Slot Isolation: Cut-in specifying distance (e.g. 'haan 42 km') only updates distance; temperature is untouched.
        """
        if self.state != DialogueState.CONFIRMING:
            raise RuntimeError(f"Confirmation replies can only be processed from CONFIRMING state, currently: {self.state}")

        self.last_reply_audio_path = reply_audio_path or audio_path
        self.last_reply_obj = reply

        raw_reply = (reply.raw_text or "").strip()
        is_silence = (raw_reply in ("[SILENCE]", "[NO_SPEECH]")) or (
            not raw_reply and (reply.intent is None or reply.intent.upper() in ("", "UNCLEAR", "SILENCE"))
        )
        if is_silence:
            return self.handle_silence_or_timeout()

        self.silence_timeout_count = 0  # Reset on valid driver response
        intent = reply.intent.upper()

        # ACOUSTIC ECHO LOOP DEFENSE:
        # Prevent prompt tails (e.g. "Sahi hai na Bhaiya?") reflected into microphone
        # or full readback reflections from self-confirming.
        if self._is_acoustic_echo(raw_reply):
            self._record_transition(
                DialogueState.CONFIRMING,
                trigger="ACOUSTIC_ECHO_REJECTED",
                payload={"raw_reply": raw_reply, "last_agent_prompt": self.last_agent_prompt}
            )
            prompt = "Echo detect hua hai. Kripya 'haan' ya 'naa' saaf bole."
            self.last_agent_prompt = prompt
            return DialogueState.CONFIRMING, prompt

        # BARGE-IN SAFETY GUARDRAIL & SLOT-ISOLATED CUT-IN:
        min_listen_time = getattr(self, "critical_slots_duration_s", getattr(self, "minimum_readback_listen_time_s", 2.2))
        if intent == "CONFIRM" and playback_elapsed_s is not None and playback_elapsed_s < min_listen_time:
            import re
            from gnani.client import parse_hindi_number
            raw_text = reply.raw_text or ""
            raw_lower = raw_text.lower()
            is_dist_cutin = bool(re.search(r"\b(km|kilometer|kilo\s*meter|k\.m\.)\b", raw_lower))
            parsed_num = parse_hindi_number(raw_text)

            if is_dist_cutin:
                expected_dist = self.current_record.distance_km if self.current_record else None
                if parsed_num is not None and expected_dist is not None and abs(parsed_num - expected_dist) >= 0.1:
                    logger.info(f"Driver cut in early with divergent distance {parsed_num} km vs read-back {expected_dist} km. Routing to CORRECT (temperature untouched).")
                    reply.intent = "CORRECT"
                    reply.corrected_distance_km = parsed_num
                    reply.corrected_temperature_c = None
                    intent = "CORRECT"
                else:
                    has_explicit_number = bool(
                        parsed_num is not None and expected_dist is not None and abs(parsed_num - expected_dist) < 0.1
                    )
                    if not has_explicit_number:
                        self._record_transition(
                            DialogueState.CONFIRMING,
                            trigger="EARLY_BARGE_IN_REJECTED",
                            payload={"elapsed_s": playback_elapsed_s, "min_listen_time": min_listen_time, "raw_reply": reply.raw_text}
                        )
                        dist_str = f"{expected_dist:g} km" if expected_dist is not None else "distance"
                        prompt = f"Bhaiya poora suniye: Distance {dist_str} bataya tha. Kya yeh sahi hai? Haan ya naa saaf bole."
                        self.last_agent_prompt = prompt
                        return DialogueState.CONFIRMING, prompt
            else:
                expected_temp = self.current_record.temperature_c if self.current_record else None
                if parsed_num is not None and expected_temp is not None and abs(parsed_num - expected_temp) >= 0.1:
                    logger.info(f"Driver cut in early with divergent number {parsed_num}°C against read-back {expected_temp}°C. Routing to CORRECT (distance untouched).")
                    reply.intent = "CORRECT"
                    reply.corrected_temperature_c = parsed_num
                    reply.corrected_distance_km = None
                    intent = "CORRECT"
                else:
                    has_explicit_number = bool(
                        parsed_num is not None and expected_temp is not None and abs(parsed_num - expected_temp) < 0.1
                    )
                    if not has_explicit_number:
                        self._record_transition(
                            DialogueState.CONFIRMING,
                            trigger="EARLY_BARGE_IN_REJECTED",
                            payload={"elapsed_s": playback_elapsed_s, "min_listen_time": min_listen_time, "raw_reply": reply.raw_text}
                        )
                        temp_str = f"{expected_temp:g} degree" if expected_temp is not None else "report"
                        prompt = f"Bhaiya poora suniye: Temperature {temp_str} bataya tha. Kya yeh sahi hai? Haan ya naa saaf bole."
                        self.last_agent_prompt = prompt
                        return DialogueState.CONFIRMING, prompt

        # CASE 1: DRIVER CONFIRMS -> EVALUATE VIA PURE GATE
        if intent == "CONFIRM":
            gate_res = evaluate_confirmation_gate(
                self.current_record,
                reply,
                self.veh_telemetry,
                is_already_challenged=getattr(self, "requires_forced_confirmation", False)
            )

            if gate_res.decision == GateDecision.CHALLENGE:
                self.requires_forced_confirmation = True
                retry_prompt = gate_res.challenge_prompt or "Kripya number repeat karein."
                self.last_agent_prompt = retry_prompt
                return DialogueState.CONFIRMING, retry_prompt

            if gate_res.decision == GateDecision.ALLOW:
                self.requires_forced_confirmation = False
                self.last_gate_residual_risk = gate_res.residual_risk_flag
                self._record_transition(
                    DialogueState.ACTING,
                    trigger="DRIVER_AFFIRMATIVE_CONFIRMATION",
                    payload={
                        "driver_reply": reply.model_dump(),
                        "record": self.current_record.model_dump(),
                        "gate_decision": gate_res.model_dump()
                    }
                )
                # Execute gated tools
                return self._execute_dispatch_action()

        # CASE 2: DRIVER CORRECTS OR REJECTS
        self.correction_rounds += 1
        
        if self.correction_rounds > self.max_correction_rounds:
            # Exceeded safety retry threshold -> Escalate to persistent supervisor queue
            self.requires_forced_confirmation = False
            self._record_transition(
                DialogueState.ESCALATING,
                trigger="EXCEEDED_MAX_CORRECTION_ROUNDS",
                payload={"rounds": self.correction_rounds, "last_reply": reply.model_dump()}
            )
            self._check_and_enqueue_escalation(trigger_reason="EXCEEDED_MAX_CORRECTION_ROUNDS")
            prompt = (
                "Bhaiya, do baar confusion hua hai. Main call turant human fleet supervisor ko "
                "connect kar raha hoon. Kripya line par bane rahein."
            )
            self.last_agent_prompt = prompt
            return DialogueState.ESCALATING, prompt

        if intent == "CORRECT":
            self._record_transition(
                DialogueState.CORRECTING,
                trigger="DRIVER_CORRECTION_REQUESTED",
                payload={"correction": reply.model_dump(), "round": self.correction_rounds}
            )
            
            orig_slots = self.current_record.model_dump()

            # Apply corrections to the current record with strict reefer physical bounds (-40C to +50C)
            if reply.corrected_temperature_c is not None:
                if -40.0 <= reply.corrected_temperature_c <= 50.0:
                    self.current_record.temperature_c = reply.corrected_temperature_c
                    self.current_record.anomalous_reading = None
                    # Update forced confirmation status if corrected temperature is now plausible
                    if self.veh_telemetry:
                        tol = get_cargo_tolerance(self.veh_telemetry.cargo_type)
                        if abs(self.current_record.temperature_c - self.veh_telemetry.target_temp_c) <= tol:
                            self.requires_forced_confirmation = False
                else:
                    logger.warning(f"Driver correction {reply.corrected_temperature_c}°C out of realistic bounds (-40 to 50°C)")
                    self.current_record.anomalous_reading = reply.corrected_temperature_c
                    self.current_record.temperature_c = None

            if reply.corrected_location:
                self.current_record.location_text = reply.corrected_location
            if reply.corrected_distance_km is not None:
                self.current_record.distance_km = reply.corrected_distance_km

            # Offline Acoustic Confusion Audit Queue: Log driver correction for human review
            try:
                from core.learning import record_driver_correction
                record_driver_correction(
                    raw_transcript=self.last_driver_utterance,
                    original_extraction=orig_slots,
                    driver_correction=reply.raw_text,
                    corrected_slots=self.current_record.model_dump(),
                    vehicle_id=self.vehicle_id
                )
            except Exception as e:
                logger.error(f"Failed to persist driver correction to audit queue: {e}")

            # Check if correction rendered record non-actionable (e.g. invalid temperature became None)
            if not self.current_record.is_actionable():
                self.requires_forced_confirmation = False
                missing = self.current_record.compute_missing_fields()
                self._record_transition(
                    DialogueState.ASKING,
                    trigger="CORRECTION_RENDERED_RECORD_INACTIONABLE",
                    payload={"missing_fields": missing, "record": self.current_record.model_dump()}
                )
                prompt = self._generate_clarification_prompt(missing)
                self.last_agent_prompt = prompt
                return self.state, prompt

            # Move back to CONFIRMING with updated values
            self._record_transition(
                DialogueState.CONFIRMING,
                trigger="READBACK_AFTER_CORRECTION",
                payload={"updated_record": self.current_record.model_dump()}
            )
            prompt = f"Theek hai, correct kar liya: {self.generate_readback_prompt(self.current_record)}"
            self.last_agent_prompt = prompt
            return self.state, prompt

        # CASE 3: UNCLEAR OR REJECT WITHOUT SPECIFIC CORRECTION
        self._record_transition(
            DialogueState.CORRECTING,
            trigger="DRIVER_UNCLEAR_OR_REJECT",
            payload={"reply": reply.model_dump()}
        )
        self._record_transition(
            DialogueState.CONFIRMING,
            trigger="RETRY_READBACK",
            payload={"record": self.current_record.model_dump()}
        )
        prompt = (
            f"Kripya dobara batayein: {self.generate_readback_prompt(self.current_record)} "
            "Haan bole agar sahi hai, ya sahi number batayein."
        )
        self.last_agent_prompt = prompt
        return self.state, prompt

    def _execute_dispatch_action(self) -> Tuple[DialogueState, str]:
        """
        THE CORE SAFETY GUARANTEE:
        This method executes ONLY if self.state == DialogueState.ACTING.
        Calls simulated fleet database and creates an emergency dispatch ticket.
        """
        if self.state != DialogueState.ACTING:
            raise PermissionError(
                f"SAFETY GATE VIOLATION: Dispatch action rejected! State is {self.state}, not ACTING."
            )

        rec = self.current_record
        if not rec:
            raise ValueError("No valid record to dispatch")

        # Physical cargo thermal excursion check independent of ASR plausibility
        veh = self.veh_telemetry or lookup_vehicle(rec.vehicle_id or self.vehicle_id)
        from core.schemas import check_cargo_temperature_excursion
        is_excursion, excursion_msg, excursion_prio = check_cargo_temperature_excursion(
            veh.cargo_type if veh else None,
            rec.temperature_c
        )
        self.confirmed_excursion = (is_excursion, excursion_msg, excursion_prio)
        self.priority_source = "CONFIRMED_READING"

        # Check if gate had residual risk (e.g. override confirmed on divergent reading)
        residual = getattr(self, "last_gate_residual_risk", False)
        final_priority = DispatchPriority.CRITICAL if (residual or (is_excursion and excursion_prio == DispatchPriority.CRITICAL)) else (excursion_prio if is_excursion else DispatchPriority.HIGH)

        notes_parts = [f"Confirmed via driver voice read-back. Raw: '{rec.raw_transcript}'"]
        if is_excursion:
            notes_parts.append(f"[EXCURSION] {excursion_msg}")
        if residual:
            notes_parts.append("[RESIDUAL RISK] Driver confirmed divergent telematics override")

        gps_lat = veh.gps_lat if veh else None
        gps_lng = veh.gps_lng if veh else None

        # Execute simulated tool
        dispatch_result = dispatch_assistance(
            vehicle_id=rec.vehicle_id or self.vehicle_id,
            incident_type=rec.incident_type.value,
            location_reported=rec.location_text or "Highway",
            distance_km=rec.distance_km,
            reported_temp_c=rec.temperature_c,
            priority=final_priority,
            notes="; ".join(notes_parts),
            supervisor_alert_flag=residual or (is_excursion and final_priority == DispatchPriority.CRITICAL),
            residual_risk_flag=residual,
            gps_lat=gps_lat,
            gps_lng=gps_lng,
            priority_source="CONFIRMED_READING"
        )
        self.last_action_result = dispatch_result

        # Store persistent forensic dispute audit bundle
        try:
            record_dispute_audit(
                ticket_id=dispatch_result["ticket_id"],
                vehicle_id=rec.vehicle_id or self.vehicle_id or "UNKNOWN",
                initial_audio_path=getattr(self, "last_audio_path", None),
                asr_transcript=self.last_driver_utterance,
                readback_prompt_spoken=self.last_agent_prompt,
                driver_reply_audio_path=getattr(self, "last_reply_audio_path", None),
                driver_reply_intent=getattr(self.last_reply_obj, "intent", "CONFIRM"),
                driver_reply_text=getattr(self.last_reply_obj, "raw_text", ""),
                gate_decision="ALLOW",
                residual_risk_flag=residual,
                priority_source="CONFIRMED_READING"
            )
        except Exception as e:
            logger.warning(f"Could not write dispute audit record: {e}")

        # Dynamically compute destination and delay from vehicle telematics
        destination_hub = "Delhi Central Depot"
        if veh and " to " in veh.route:
            destination_hub = f"{veh.route.split(' to ')[-1].strip()} Hub"

        est_delay = max(30, int(dispatch_result.get("technician_eta_min", 45)) + 15)

        update_eta(
            vehicle_id=rec.vehicle_id or self.vehicle_id,
            destination=destination_hub,
            delay_minutes=est_delay,
            reason=f"Incident: {rec.incident_type.value} at {rec.location_text}"
        )

        sup_note = " Fleet supervisor ko bhi alert bhej diya gaya hai." if residual else ""
        spoken_confirmation = (
            f"Dhanyawad Bhaiya! Assistance ticket {dispatch_result['ticket_id']} dispatch ho gaya hai. "
            f"Technician lagbhag {dispatch_result['technician_eta_min']} minute mein pahunch rahe hain.{sup_note} "
            f"Safe rahein!"
        )
        self.last_agent_prompt = spoken_confirmation

        # Transition to DONE
        self._record_transition(
            DialogueState.DONE,
            trigger="DISPATCH_ACTION_EXECUTED_SUCCESSFULLY",
            payload={"result": dispatch_result}
        )

        # Upgrade prior unverified supervisor escalation on confirmed dispatch
        try:
            from core.tools import upgrade_supervisor_escalation_on_confirmation
            upgrade_supervisor_escalation_on_confirmation(
                vehicle_id=rec.vehicle_id or self.vehicle_id,
                confirmed_ticket_id=dispatch_result["ticket_id"],
                confirmed_temp_c=rec.temperature_c
            )
        except Exception as e:
            logger.warning(f"Could not upgrade supervisor escalation on confirmation: {e}")

        return DialogueState.DONE, spoken_confirmation

    def _generate_clarification_prompt(self, missing_fields: List[str]) -> str:
        if self.current_record and getattr(self.current_record, "anomalous_reading", None) is not None:
            val = int(self.current_record.anomalous_reading)
            return f"Bhaiya, aapne {val} degree bola jo unusual hai. Kya sahi temperature 18 degree hai ya kuch aur?"
        if "temperature_c" in missing_fields and "location_text" in missing_fields:
            return "Bhaiya, reefer ka temperature kitna dikha raha hai aur gaadi abhi kahan par hai?"
        elif "temperature_c" in missing_fields:
            return "Bhaiya, temperature kitna ho gaya hai reefer ka?"
        elif "location_text" in missing_fields:
            return "Bhaiya, aap abhi highway par kis landmark ya toll ke paas hain?"
        return "Bhaiya, kripya thoda aur detail batayein."

    def resolve_escalation_by_supervisor(self, supervisor_id: str = "TOWER_SUP_01", action_taken: str = "MANUAL_DISPATCH") -> Tuple[DialogueState, str]:
        """Supervisory takeover: transitions ESCALATING state to terminal DONE with audit trail."""
        if self.state != DialogueState.ESCALATING:
            raise RuntimeError(f"Can only resolve escalation from ESCALATING state, currently: {self.state}")

        self.requires_forced_confirmation = False
        self._record_transition(
            DialogueState.DONE,
            trigger="SUPERVISOR_RESOLUTION",
            payload={"supervisor_id": supervisor_id, "action": action_taken}
        )
        msg = f"Escalation successfully resolved by supervisor {supervisor_id}. Action: {action_taken}"
        self.last_agent_prompt = msg
        return DialogueState.DONE, msg

    def reset(self, new_vehicle_id: Optional[str] = None):
        """Reset session to LISTENING state for new conversation."""
        if new_vehicle_id:
            self.vehicle_id = new_vehicle_id
        self.veh_telemetry = lookup_vehicle(self.vehicle_id) if self.vehicle_id else None
        self.is_unverified_vehicle = (self.veh_telemetry is None)
        self.requires_forced_confirmation = False
        self._record_transition(DialogueState.LISTENING, "SESSION_RESET", {"vehicle_id": self.vehicle_id})
        self.current_record = None
        self.last_driver_utterance = ""
        self.last_agent_prompt = ""
        self.last_action_result = None
        self.correction_rounds = 0
