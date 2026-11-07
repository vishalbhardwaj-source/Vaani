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
    SENSOR_MATCH_TOLERANCE_C,
    get_cargo_tolerance,
    GateDecision,
    GateResult,
    DispatchPriority
)
from core.tools import dispatch_assistance, update_eta, lookup_vehicle, lookup_vehicle_by_phone

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
    sensor_match_tolerance = SENSOR_MATCH_TOLERANCE_C
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
    # Covers:
    # A) Target is negative, spoken is positive (e.g. target -22°C, spoken +14°C or +18°C)
    # B) Target is positive, spoken is negative (e.g. target +4°C, spoken -4°C or -14°C)
    # C) Magnitude matches live sensor with opposite sign (e.g. sensor -14°C, spoken +14°C - dropped minus sign!)
    is_target_polarity_flip = (target_c < 0.0 and record.temperature_c > 0.0) or (target_c > 0.0 and record.temperature_c < 0.0)
    is_sensor_magnitude_sign_flip = (
        sensor_is_valid and sensor_c != 0.0 and
        (record.temperature_c * sensor_c < 0.0) and
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


class DialogueController:
    """
    Finite State Machine governing driver interactions and action gate.
    All transitions are recorded with precise audit trails.
    """
    def __init__(
        self,
        vehicle_id: Optional[str] = "HR-55-TC-1102",
        caller_phone: Optional[str] = None,
        max_correction_rounds: int = 2
    ):
        self.state: DialogueState = DialogueState.LISTENING
        self.caller_phone = caller_phone
        self.max_correction_rounds = max_correction_rounds
        self.correction_rounds = 0
        self.requires_forced_confirmation = False
        self.last_gate_residual_risk = False
        self.registration_retries = 0
        self.max_registration_retries = 2
        self.unconfirmed_excursion: Optional[Tuple[bool, str, DispatchPriority]] = None
        self.confirmed_excursion: Optional[Tuple[bool, str, DispatchPriority]] = None
        self.priority_source: str = "NONE"

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

    def receive_driver_utterance(self, transcript: str) -> Tuple[DialogueState, str]:
        """
        Step 1: Driver speaks in LISTENING state.
        Transitions: LISTENING -> EXTRACTING
        """
        if self.state not in (DialogueState.LISTENING, DialogueState.DONE):
            raise RuntimeError(f"Cannot receive initial utterance while in state {self.state}")

        self.last_driver_utterance = transcript.strip()
        self.correction_rounds = 0
        self.last_action_result = None
        
        self._record_transition(
            DialogueState.EXTRACTING,
            trigger="DRIVER_INITIAL_REPORT",
            payload={"raw_transcript": self.last_driver_utterance}
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

        # Telematics location anchor: If spoken location is missing or generic, anchor to telematics GPS
        if not record.location_text:
            if self.veh_telemetry and getattr(self.veh_telemetry, "last_known_location", None):
                record.location_text = f"GPS: {self.veh_telemetry.last_known_location}"

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
        Enforces a retry cap (default 2) to prevent unidentifiable drivers from getting trapped in loops.
        """
        from core.tools import lookup_vehicle_by_registration_query
        veh, match_status = lookup_vehicle_by_registration_query(reply_text)
        if veh:
            self.vehicle_id = veh.vehicle_id
            self.veh_telemetry = veh
            self.is_unverified_vehicle = False
            self.unverified_reason = None
            self.registration_retries = 0
            self._record_transition(
                DialogueState.EXTRACTING,
                trigger="VEHICLE_IDENTIFIED",
                payload={"vehicle_id": veh.vehicle_id, "status": match_status}
            )
            if self.current_record:
                self.current_record.vehicle_id = veh.vehicle_id
                return self.process_extraction(self.current_record)
            else:
                prompt = f"Gaadi {veh.vehicle_id} verify ho gayi hai. Batayein kya samasya hai?"
                self.last_agent_prompt = prompt
                return DialogueState.LISTENING, prompt
        else:
            self.registration_retries += 1
            if self.registration_retries >= self.max_registration_retries:
                self._record_transition(
                    DialogueState.ESCALATING,
                    trigger="EXCEEDED_MAX_REGISTRATION_RETRIES",
                    payload={"attempts": self.registration_retries, "last_query": reply_text}
                )
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

    def generate_readback_prompt(self, record: ExtractionRecord, force_digit_spelling: bool = False) -> str:
        """
        Generate spoken Hinglish read-back string for Timbre TTS.
        Normal read-back is concise (~3.5 s). Digit-by-digit spelling is reserved for
        when the telematics gate challenges an anomalous reading or in extreme noise conditions.
        Supports full barge-in / interruptibility during playback.
        Example normal: 'Temperature 18 degree, toll plaza se 42 km aage. Sahi hai na Bhaiya?'
        Example challenged: 'Temperature 80 degree (8, 0), toll plaza se 42 (4, 2) km aage. Dhyan dein: ...'
        """
        use_spelling = force_digit_spelling or getattr(self, "requires_forced_confirmation", False)

        parts = []
        if record.temperature_c is not None:
            temp_num = int(record.temperature_c) if record.temperature_c.is_integer() else record.temperature_c
            sign_word = "minus " if record.temperature_c < 0 else ""
            if use_spelling:
                digits = ", ".join(list(str(abs(int(temp_num)))))
                parts.append(f"Temperature {sign_word}{temp_num} degree ({sign_word}{digits})")
            else:
                parts.append(f"Temperature {sign_word}{temp_num} degree")

        if record.location_text:
            if record.location_text.startswith("GPS: "):
                age_str = f" ({self.veh_telemetry.gps_fix_age_minutes} minute purana)" if (self.veh_telemetry and getattr(self.veh_telemetry, "gps_fix_age_minutes", None)) else ""
                parts.append(f"GPS telemetry{age_str} ke mutabik {record.location_text[5:]}")
            elif record.distance_km is not None:
                dist_num = int(record.distance_km) if record.distance_km.is_integer() else record.distance_km
                if use_spelling:
                    dist_digits = ", ".join(list(str(abs(int(dist_num)))))
                    parts.append(f"{record.location_text} se {dist_num} ({dist_digits}) km aage")
                else:
                    parts.append(f"{record.location_text} se {dist_num} km aage")
            else:
                parts.append(f"{record.location_text} ke paas")

        if record.incident_type == IncidentType.BREAKDOWN:
            parts.insert(0, f"Vehicle {self.vehicle_id} breakdown report")

        # Telematics plausibility policy: Flag deviation beyond per-cargo class tolerance
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
                parts.append(f"Dhyan dein: {comp} aur report {int(record.temperature_c)} degree hai")

        details = ", ".join(parts) if parts else "Diya gaya report"
        return f"{details}. Sahi hai na Bhaiya?"

    def handle_confirmation_reply(self, reply: DriverReply) -> Tuple[DialogueState, str]:
        """
        Step 3: Evaluate driver reply to read-back using pure confirmation gate.
        Rules:
        - CONFIRM -> gate evaluates plausibility:
            - If ALLOW -> transitions to ACTING (and immediately executes dispatch) -> DONE
            - If CHALLENGE -> prompts for explicit number repetition
        - CORRECT -> transitions to CORRECTING -> CONFIRMING
        - REJECT / UNCLEAR -> treated as not confirmed; retries or escalates
        - If correction_rounds exceeds max_correction_rounds -> ESCALATING
        """
        if self.state != DialogueState.CONFIRMING:
            raise RuntimeError(f"Confirmation replies can only be processed from CONFIRMING state, currently: {self.state}")

        intent = reply.intent.upper()

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
            # Exceeded safety retry threshold -> Escalate to human control tower
            self.requires_forced_confirmation = False
            self._record_transition(
                DialogueState.ESCALATING,
                trigger="EXCEEDED_MAX_CORRECTION_ROUNDS",
                payload={"rounds": self.correction_rounds, "last_reply": reply.model_dump()}
            )
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
