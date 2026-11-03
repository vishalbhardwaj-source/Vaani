"""
Core module for Vaani dialogue controller, schemas, and simulated tools.
"""
from core.schemas import DialogueState, ExtractionRecord, DriverReply, IncidentType
from core.state_machine import DialogueController
from core.tools import lookup_vehicle, dispatch_assistance, update_eta, get_action_log, init_db
from core.learning import (
    record_driver_correction,
    get_learned_corrections,
    get_adaptive_rules,
    get_few_shot_prompt_context,
    synthesize_new_rules,
    init_learning_db
)

__all__ = [
    "DialogueState",
    "ExtractionRecord",
    "DriverReply",
    "IncidentType",
    "DialogueController",
    "lookup_vehicle",
    "dispatch_assistance",
    "update_eta",
    "get_action_log",
    "init_db",
    "record_driver_correction",
    "get_learned_corrections",
    "get_adaptive_rules",
    "get_few_shot_prompt_context",
    "synthesize_new_rules",
    "init_learning_db"
]
