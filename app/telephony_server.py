"""
Vaani: Production Telephony Gateway & WebSocket Streaming Server
Connects real telephony trunks (SIP / Twilio Media Streams / Exotel / WhatsApp Cloud Calling)
to the Vaani Dialogue Controller, Whisper-LoRA ASR, and Timbre TTS.

Key Capabilities:
1. Full-Duplex WebSocket Audio Streaming (/telephony/stream)
2. Energy-based Voice Activity Detection (VAD) & Frame-Level Turn-Taking
3. Barge-In Interruption: Cancels outgoing TTS playback when driver speaks
4. Silence / Timeout State Machine Trigger (Reprompts -> Escalates)
5. Live Supervisor Queue API & Forensic Dispute Audit Retrieval
"""

import sys
import io
import json
import base64
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    try:
        import audioop
    except ImportError:
        audioop = None

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.schemas import DialogueState, ExtractionRecord, DriverReply, IncidentType
from core.state_machine import DialogueController
from core.tools import (
    list_supervisor_queue,
    resolve_supervisor_escalation,
    get_dispute_audit,
    get_action_log,
    init_db
)
from gnani.client import extract_incident_details, classify_driver_reply, speak_text

logger = logging.getLogger("vaani.telephony")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(
    title="Vaani Telephony & Streaming Audio Server",
    description="Live SIP/WebRTC and CPaaS Bridge for Voice-Assisted Reefer Telematics",
    version="2.0.0"
)


# -------------------------------------------------------------
# 1. Audio Transcoding & Energy-Based VAD Utilities
# -------------------------------------------------------------
def decode_mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    """Converts 8 kHz G.711 mu-law telephony audio to 16 kHz Linear PCM."""
    try:
        pcm8 = audioop.ulaw2lin(mulaw_bytes, 2)
        pcm16, _ = audioop.ratecv(pcm8, 2, 1, 8000, 16000, None)
        return pcm16
    except Exception:
        # Fallback if audioop is not available
        return mulaw_bytes


def compute_frame_energy(pcm_bytes: bytes) -> float:
    """Calculates RMS energy of 16-bit linear PCM audio frame."""
    if not pcm_bytes:
        return 0.0
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0


# -------------------------------------------------------------
# 2. WebSocket Streaming Telephony Bridge
# -------------------------------------------------------------
@app.websocket("/telephony/stream")
async def telephony_audio_stream(
    websocket: WebSocket,
    caller_phone: Optional[str] = Query(None),
    vehicle_id: Optional[str] = Query(None)
):
    """
    Full-duplex WebSocket stream compatible with Twilio Media Streams, Exotel,
    and FreeSWITCH / Asterisk ARI.
    Handles:
    - Inbound raw audio packet streaming (G.711 mu-law or Linear PCM)
    - Real-time VAD speech framing
    - Barge-in playback cancellation
    - Silence timeout detection and reprompting
    """
    await websocket.accept()
    logger.info(f"[Telephony] Inbound call established. Caller: {caller_phone}, Vehicle: {vehicle_id}")

    # Initialize Dialogue Controller for this active call session
    controller = DialogueController(vehicle_id=vehicle_id, caller_phone=caller_phone)
    init_db()

    # VAD Framing State
    VAD_ENERGY_THRESHOLD = 0.015   # RMS floor above background road rumble
    SILENCE_HANG_FRAMES = 25       # 25 frames * 20ms = 500ms trailing silence to finalize turn
    FRAME_INTERVAL_SEC = 0.020     # 20ms telephony packet size

    is_speaking = False
    silence_frames_count = 0
    speech_buffer = bytearray()
    last_speech_time = time.time()
    agent_is_playing_tts = False
    tts_playback_started_at = None
    barge_in_elapsed_s = None

    # Send initial greeting
    greeting_text = (
        f"Namaste Bhaiya, main Vaani cold-chain assistant hoon. "
        f"Gaadi {controller.vehicle_id or 'telematics'} linked hai. Batayein kya samasya aayi hai?"
    )
    await websocket.send_json({
        "event": "agent_prompt",
        "state": controller.state.value,
        "text": greeting_text,
        "audio_base64": base64.b64encode(speak_text(greeting_text)).decode("utf-8")
    })
    last_speech_time = time.time()
    hangup_handled = False

    try:
        while True:
            raw_msg = await websocket.receive_text()
            data = json.loads(raw_msg)
            event_type = data.get("event")

            # 1. Handle incoming audio payload (Twilio 'media' / Exotel 'stream' format)
            if event_type in ("media", "audio_frame"):
                payload_b64 = data.get("media", {}).get("payload") or data.get("audio_payload")
                if not payload_b64:
                    continue

                raw_bytes = base64.b64decode(payload_b64)
                # Convert mu-law if format is G.711
                encoding = data.get("encoding", "mulaw")
                pcm_frame = decode_mulaw_to_pcm16(raw_bytes) if encoding == "mulaw" else raw_bytes
                energy = compute_frame_energy(pcm_frame)

                # BARGE-IN INTERRUPTION: If agent is speaking and driver interrupts
                if agent_is_playing_tts and energy > (VAD_ENERGY_THRESHOLD * 1.5):
                    logger.info("[Telephony] Barge-in detected: Driver interrupted agent playback.")
                    if tts_playback_started_at:
                        barge_in_elapsed_s = time.time() - tts_playback_started_at
                    agent_is_playing_tts = False
                    await websocket.send_json({
                        "event": "playback_cancelled",
                        "reason": "DRIVER_BARGE_IN",
                        "elapsed_s": barge_in_elapsed_s
                    })

                # VAD SPEECH ONSET
                if energy >= VAD_ENERGY_THRESHOLD:
                    is_speaking = True
                    silence_frames_count = 0
                    last_speech_time = time.time()
                    speech_buffer.extend(pcm_frame)
                elif is_speaking:
                    # Speech continuation during short pause
                    speech_buffer.extend(pcm_frame)
                    silence_frames_count += 1

                    # VAD SPEECH OFFSET: User finished their turn
                    if silence_frames_count >= SILENCE_HANG_FRAMES:
                        is_speaking = False
                        silence_frames_count = 0
                        logger.info(f"[Telephony] Speech turn completed. Total buffer: {len(speech_buffer)} bytes")

                        # Transcribe accumulated speech chunk
                        turn_audio_bytes = bytes(speech_buffer)
                        speech_buffer.clear()

                        # Pass through Whisper-LoRA ASR
                        try:
                            from core.asr_model import get_asr_service
                            asr_svc = get_asr_service()
                            asr_res = asr_svc.transcribe(turn_audio_bytes)
                            transcript = asr_res["transcript"]
                            confidence = asr_res["confidence"]
                        except Exception as e:
                            logger.warning(f"Whisper ASR inference fallback: {e}")
                            transcript = ""
                            confidence = 0.50

                        logger.info(f"[Telephony] ASR Transcript: '{transcript}' (Confidence: {confidence})")

                        # Process with Dialogue State Machine
                        if controller.state in (DialogueState.LISTENING, DialogueState.EXTRACTING):
                            state, agent_msg = controller.receive_driver_utterance(
                                transcript,
                                asr_confidence=confidence
                            )
                            if state == DialogueState.EXTRACTING:
                                rec = extract_incident_details(transcript)
                                state, agent_msg = controller.process_extraction(rec)
                        elif controller.state == DialogueState.CONFIRMING:
                            reply = classify_driver_reply(transcript)
                            state, agent_msg = controller.handle_confirmation_reply(
                                reply,
                                playback_elapsed_s=barge_in_elapsed_s
                            )
                            barge_in_elapsed_s = None
                        elif controller.state == DialogueState.ASKING:
                            if controller.is_unverified_vehicle:
                                state, agent_msg = controller.receive_vehicle_identification_reply(transcript)
                            else:
                                rec_upd = extract_incident_details(transcript)
                                state, agent_msg = controller.receive_missing_slot_reply(transcript, rec_upd)
                        else:
                            state = controller.state
                            agent_msg = controller.last_agent_prompt

                        # Synthesize response audio
                        agent_is_playing_tts = True
                        tts_playback_started_at = time.time()
                        barge_in_elapsed_s = None
                        tts_bytes = speak_text(agent_msg)
                        await websocket.send_json({
                            "event": "agent_prompt",
                            "state": state.value,
                            "text": agent_msg,
                            "transcript_recognized": transcript,
                            "asr_confidence": confidence,
                            "audio_base64": base64.b64encode(tts_bytes).decode("utf-8")
                        })
                        last_speech_time = time.time()

            # 2. Check for Silence Timeout when driver says nothing
            now = time.time()
            if not is_speaking and (now - last_speech_time > 4.5):
                logger.info("[Telephony] Silence timeout threshold reached. Triggering reprompt.")
                state, timeout_prompt = controller.handle_silence_or_timeout()
                tts_bytes = speak_text(timeout_prompt)
                agent_is_playing_tts = True
                await websocket.send_json({
                    "event": "agent_prompt",
                    "state": state.value,
                    "text": timeout_prompt,
                    "trigger": "SILENCE_TIMEOUT",
                    "audio_base64": base64.b64encode(tts_bytes).decode("utf-8")
                })
                last_speech_time = time.time()

            # 3. Call Hangup
            if event_type == "stop":
                logger.info("[Telephony] Call hung up by remote party.")
                break

    except WebSocketDisconnect:
        logger.info("[Telephony] Client disconnected from WebSocket stream.")
    except Exception as e:
        logger.error(f"[Telephony] Error during stream processing: {e}")
    finally:
        # Check if call was dropped with an unconfirmed sensor-corroborated excursion
        # Guarded to run strictly once and catch internal exceptions so socket cleanup is never masked
        if not hangup_handled:
            hangup_handled = True
            try:
                if controller.state not in (DialogueState.DONE, DialogueState.ESCALATING):
                    logger.info("[Telephony] Remote call dropped before completion. Running fail-safe check.")
                    controller._check_and_enqueue_escalation(trigger_reason="TELEPHONY_CALL_DROPPED_BEFORE_CONFIRMATION")
            except Exception as e:
                logger.error(f"[Telephony] Error during call hangup fail-safe: {e}")


# -------------------------------------------------------------
# 3. Supervisor Queue & Dispute Audit REST Endpoints
# -------------------------------------------------------------
class ResolutionRequest(BaseModel):
    escalation_id: int
    resolution_notes: str


@app.get("/api/health")
def health_check():
    return {"status": "ONLINE", "streaming": "active", "service": "Vaani Telephony Gateway", "version": "2.0.0"}


@app.get("/api/supervisor/queue")
def get_pending_supervisor_queue(status: Optional[str] = "PENDING"):
    """Fetches real-time escalation tickets awaiting fleet supervisor action."""
    tickets = list_supervisor_queue(status=status)
    return {"count": len(tickets), "tickets": tickets}


@app.post("/api/supervisor/resolve")
def resolve_ticket(req: ResolutionRequest):
    """Marks an escalation ticket resolved with human notes."""
    success = resolve_supervisor_escalation(req.escalation_id, req.resolution_notes)
    if not success:
        raise HTTPException(status_code=404, detail=f"Ticket {req.escalation_id} not found")
    return {"status": "RESOLVED", "escalation_id": req.escalation_id}


@app.get("/api/audit/dispute/{ticket_id}")
def get_forensic_dispute_bundle(ticket_id: int):
    """Retrieves full dispute audit bundle for a disputed dispatch."""
    audit = get_dispute_audit(ticket_id)
    if not audit:
        raise HTTPException(status_code=404, detail=f"No forensic audit found for ticket {ticket_id}")
    return audit
