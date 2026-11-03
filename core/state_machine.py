r"""
Vaani: Dialogue Controller & Finite State Machine
Guarantees the safety gate: Dispatch actions CAN NEVER run without explicit read-back confirmation.

States:
LISTENING -> EXTRACTING -> (ASKING) -> CONFIRMING -> (CORRECTING) -> ACTING -> DONE
                                                    \-> ESCALATING -> DONE
"""

import datetime
from typing import Optional, List, Dict, Any, Tuple
from core.schemas import (
    DialogueState,
    ExtractionRecord,
    DriverReply,
    IncidentType,
    StateTransitionLog
)
from core.tools import dispatch_assistance, update_eta, lookup_vehicle


class DialogueController:
    """
    Finite State Machine governing driver interactions and action gate.
    All transitions are recorded with precise audit trails.
    """
    def __init__(self, vehicle_id: str = "HR-55-TC-1102", max_correction_rounds: int = 2):
        self.state: DialogueState = DialogueState.LISTENING
        self.vehicle_id = vehicle_id
        self.max_correction_rounds = max_correction_rounds
        self.correction_rounds = 0
        
        self.current_record: Optional[ExtractionRecord] = None
        self.last_driver_utterance: str = ""
        self.last_agent_prompt: str = ""
        self.last_action_result: Optional[Dict[str, Any]] = None
        
        self.history: List[StateTransitionLog] = []
        self._record_transition(DialogueState.LISTENING, "SESSION_INITIALIZED", {})

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

        # Validate vehicle ID fallback
        if not record.vehicle_id:
            record.vehicle_id = self.vehicle_id
            
        self.current_record = record

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

    def receive_missing_slot_reply(self, answer_transcript: str, updated_record: ExtractionRecord) -> Tuple[DialogueState, str]:
        """
        Handle driver response to clarification question while in ASKING state.
        Transitions: ASKING -> EXTRACTING -> process_extraction()
        """
        if self.state != DialogueState.ASKING:
            raise RuntimeError(f"Cannot process clarification reply from state {self.state}")

        self.last_driver_utterance = answer_transcript
        self._record_transition(
            DialogueState.EXTRACTING,
            trigger="DRIVER_CLARIFICATION_RECEIVED",
            payload={"answer": answer_transcript}
        )
        return self.process_extraction(updated_record)

    def generate_readback_prompt(self, record: ExtractionRecord) -> str:
        """
        Generate spoken Hinglish read-back string for Timbre TTS.
        Example: 'Temperature 28 degree, toll plaza se 42 km aage. Sahi hai?'
        """
        parts = []
        if record.temperature_c is not None:
            parts.append(f"Temperature {int(record.temperature_c) if record.temperature_c.is_integer() else record.temperature_c} degree")
            
        if record.location_text:
            if record.distance_km is not None:
                parts.append(f"{record.location_text} se {int(record.distance_km) if record.distance_km.is_integer() else record.distance_km} km aage")
            else:
                parts.append(f"{record.location_text} ke paas")
                
        if record.incident_type == IncidentType.BREAKDOWN:
            parts.insert(0, "Breakdown report")

        details = ", ".join(parts) if parts else "Diya gaya report"
        return f"{details}. Sahi hai na Bhaiya?"

    def handle_confirmation_reply(self, reply: DriverReply) -> Tuple[DialogueState, str]:
        """
        Step 3: Evaluate driver reply to read-back.
        Rules:
        - CONFIRM -> transitions to ACTING (and immediately executes dispatch) -> DONE
        - CORRECT -> transitions to CORRECTING -> CONFIRMING
        - REJECT / UNCLEAR -> treated as not confirmed; retries or escalates
        - If correction_rounds exceeds max_correction_rounds -> ESCALATING -> DONE
        """
        if self.state != DialogueState.CONFIRMING:
            raise RuntimeError(f"Confirmation replies can only be processed from CONFIRMING state, currently: {self.state}")

        intent = reply.intent.upper()

        # CASE 1: DRIVER CONFIRMS -> PROCEED TO ACTING
        if intent == "CONFIRM":
            self._record_transition(
                DialogueState.ACTING,
                trigger="DRIVER_AFFIRMATIVE_CONFIRMATION",
                payload={"driver_reply": reply.model_dump(), "record": self.current_record.model_dump()}
            )
            # Execute gated tools
            return self._execute_dispatch_action()

        # CASE 2: DRIVER CORRECTS OR REJECTS
        self.correction_rounds += 1
        
        if self.correction_rounds > self.max_correction_rounds:
            # Exceeded safety retry threshold -> Escalate to human control tower
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
            self._record_transition(
                DialogueState.DONE,
                trigger="ESCALATION_COMPLETED",
                payload={"escalated_to": "Human Control Tower"}
            )
            return DialogueState.ESCALATING, prompt

        if intent == "CORRECT":
            self._record_transition(
                DialogueState.CORRECTING,
                trigger="DRIVER_CORRECTION_REQUESTED",
                payload={"correction": reply.model_dump(), "round": self.correction_rounds}
            )
            
            orig_slots = self.current_record.model_dump()

            # Apply corrections to the current record
            if reply.corrected_temperature_c is not None:
                self.current_record.temperature_c = reply.corrected_temperature_c
            if reply.corrected_location:
                self.current_record.location_text = reply.corrected_location
            if reply.corrected_distance_km is not None:
                self.current_record.distance_km = reply.corrected_distance_km

            # Continuous Learning Loop: Persist driver correction to LLM memory
            try:
                from core.learning import record_driver_correction
                record_driver_correction(
                    raw_transcript=self.last_driver_utterance,
                    original_extraction=orig_slots,
                    driver_correction=reply.raw_text,
                    corrected_slots=self.current_record.model_dump()
                )
            except Exception:
                pass

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

        # Execute simulated tool
        dispatch_result = dispatch_assistance(
            vehicle_id=rec.vehicle_id or self.vehicle_id,
            incident_type=rec.incident_type.value,
            location_reported=rec.location_text or "Highway",
            distance_km=rec.distance_km,
            reported_temp_c=rec.temperature_c,
            notes=f"Confirmed via driver voice read-back. Raw: '{rec.raw_transcript}'"
        )
        self.last_action_result = dispatch_result

        # Update simulated ETA delay
        update_eta(
            vehicle_id=rec.vehicle_id or self.vehicle_id,
            destination="Delhi Central Depot",
            delay_minutes=60,
            reason=f"Incident: {rec.incident_type.value} at {rec.location_text}"
        )

        spoken_confirmation = (
            f"Dhanyawad Bhaiya! Assistance ticket {dispatch_result['ticket_id']} dispatch ho gaya hai. "
            f"Technician lagbhag {dispatch_result['technician_eta_min']} minute mein pahunch rahe hain. "
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
        if "temperature_c" in missing_fields and "location_text" in missing_fields:
            return "Bhaiya, reefer ka temperature kitna dikha raha hai aur gaadi abhi kahan par hai?"
        elif "temperature_c" in missing_fields:
            return "Bhaiya, temperature kitna ho gaya hai reefer ka?"
        elif "location_text" in missing_fields:
            return "Bhaiya, aap abhi highway par kis landmark ya toll ke paas hain?"
        return "Bhaiya, kripya thoda aur detail batayein."

    def reset(self):
        """Reset session to LISTENING state for new conversation."""
        self._record_transition(DialogueState.LISTENING, "SESSION_RESET", {})
        self.current_record = None
        self.last_driver_utterance = ""
        self.last_agent_prompt = ""
        self.last_action_result = None
        self.correction_rounds = 0
