"""
Vaani: Offline Acoustic Confusion Logging & Supervisor Review Queue
Captures read-back discrepancies and driver corrections for offline analytics,
error analysis, and supervised model retraining.

SAFETY ARCHITECTURE NOTE:
To prevent stored prompt injection, preserve true physical alarms (e.g. real 40°C fires),
and eliminate autonomous silent rewrites, unreviewed runtime corrections are NEVER
injected into live extraction prompts or model inference paths. All driver feedback
is stored strictly in an audit queue for offline human supervisor inspection.
"""

import sqlite3
import datetime
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "fleet.sqlite"
_learning_db_initialized = False


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_learning_db():
    """Ensure audit tables are initialized for offline acoustic review."""
    global _learning_db_initialized
    if _learning_db_initialized:
        return

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS learned_corrections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        vehicle_id TEXT,
        raw_transcript TEXT NOT NULL,
        original_extraction TEXT NOT NULL,
        driver_correction TEXT NOT NULL,
        acoustic_trap TEXT,
        corrected_slots TEXT NOT NULL,
        review_status TEXT DEFAULT 'PENDING_SUPERVISOR_AUDIT',
        reviewer_notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Migration check for existing tables
    cursor.execute("PRAGMA table_info(learned_corrections)")
    existing_cols = {row["name"] for row in cursor.fetchall()}
    if "vehicle_id" not in existing_cols:
        cursor.execute("ALTER TABLE learned_corrections ADD COLUMN vehicle_id TEXT DEFAULT 'UNKNOWN'")
    if "review_status" not in existing_cols:
        cursor.execute("ALTER TABLE learned_corrections ADD COLUMN review_status TEXT DEFAULT 'PENDING_SUPERVISOR_AUDIT'")
    if "reviewer_notes" not in existing_cols:
        cursor.execute("ALTER TABLE learned_corrections ADD COLUMN reviewer_notes TEXT")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS offline_confusion_patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern TEXT UNIQUE NOT NULL,
        observed_confusion TEXT NOT NULL,
        frequency INTEGER DEFAULT 1,
        severity TEXT DEFAULT 'MEDIUM',
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Initial reference patterns documented during fleet acoustic audits
    cursor.execute("SELECT COUNT(*) FROM offline_confusion_patterns")
    if cursor.fetchone()[0] == 0:
        seed_patterns = [
            ("18 vs 80 phonetic confusion", "Acoustic ambiguity between 18 ('atharah') and 80 ('assi') under 10dB diesel rumble.", 14, "HIGH"),
            ("14 vs 40 phonetic confusion", "Acoustic ambiguity between 14 ('choudah') and 40 ('chalis') under road noise.", 8, "HIGH"),
            ("Negative polarity drop", "Engine rumble swallowing 'minus' sign in sub-zero consignments.", 9, "CRITICAL"),
            ("Highway milestone vs temperature", "Distance milestones (e.g. 68 km) phonetically confused with temperature values.", 5, "MEDIUM"),
        ]
        cursor.executemany("""
        INSERT OR IGNORE INTO offline_confusion_patterns (pattern, observed_confusion, frequency, severity)
        VALUES (?, ?, ?, ?)
        """, seed_patterns)

    conn.commit()
    conn.close()
    _learning_db_initialized = True


def get_rules_version_hash() -> str:
    """Return MD5 hash of current audit patterns for tracking."""
    init_learning_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT pattern, updated_at FROM offline_confusion_patterns ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    content = "|".join(f"{r['pattern']}_{r['updated_at']}" for r in rows)
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:10]


def record_driver_correction(
    raw_transcript: str,
    original_extraction: Dict[str, Any],
    driver_correction: str,
    corrected_slots: Dict[str, Any],
    acoustic_trap: Optional[str] = None,
    vehicle_id: Optional[str] = None
) -> int:
    """
    Called whenever a driver corrects a read-back prompt.
    Persists the failure/correction pair strictly for offline supervisor audit.
    NEVER mutates live prompt contexts or performs autonomous rewrites.
    """
    init_learning_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    import json
    # Sanitize and truncate strings to prevent audit log bloat
    clean_raw = str(raw_transcript).strip()[:500]
    clean_driver = str(driver_correction).strip()[:500]
    orig_str = json.dumps(original_extraction, ensure_ascii=False)
    corr_str = json.dumps(corrected_slots, ensure_ascii=False)

    cursor.execute("""
    INSERT INTO learned_corrections (
        vehicle_id, raw_transcript, original_extraction, driver_correction,
        acoustic_trap, corrected_slots, review_status
    ) VALUES (?, ?, ?, ?, ?, ?, 'PENDING_SUPERVISOR_AUDIT')
    """, (vehicle_id or "UNKNOWN", clean_raw, orig_str, clean_driver, acoustic_trap or "Driver Spoken Correction", corr_str))

    rec_id = cursor.lastrowid

    # Update offline confusion frequency strictly for analytics/reporting
    pattern = None
    orig_temp = original_extraction.get("temperature_c")
    corr_temp = corrected_slots.get("temperature_c")
    
    if orig_temp is not None and corr_temp is not None:
        if abs(orig_temp - 80.0) < 0.1 and abs(corr_temp - 18.0) < 0.1:
            pattern = "18 vs 80 phonetic confusion"
        elif abs(orig_temp - 40.0) < 0.1 and abs(corr_temp - 14.0) < 0.1:
            pattern = "14 vs 40 phonetic confusion"
        elif corr_temp < 0 and orig_temp >= 0:
            pattern = "Negative polarity drop"

    if pattern:
        cursor.execute("""
        UPDATE offline_confusion_patterns 
        SET frequency = frequency + 1, updated_at = CURRENT_TIMESTAMP
        WHERE pattern = ?
        """, (pattern,))

    conn.commit()
    conn.close()
    return rec_id


def get_learned_corrections(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve driver corrections from offline audit queue."""
    init_learning_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, vehicle_id, raw_transcript, original_extraction, driver_correction, 
           acoustic_trap, corrected_slots, review_status, reviewer_notes, created_at
    FROM learned_corrections 
    ORDER BY id DESC 
    LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()

    import json
    return [
        {
            "id": r["id"],
            "vehicle_id": r["vehicle_id"],
            "raw_transcript": r["raw_transcript"],
            "original_extraction": json.loads(r["original_extraction"]),
            "driver_correction": r["driver_correction"],
            "acoustic_trap": r["acoustic_trap"],
            "corrected_slots": json.loads(r["corrected_slots"]),
            "review_status": r["review_status"],
            "reviewer_notes": r["reviewer_notes"],
            "created_at": r["created_at"]
        }
        for r in rows
    ]


def get_adaptive_rules() -> List[Dict[str, Any]]:
    """Alias for backwards compatibility: returns offline observed confusion patterns."""
    init_learning_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, pattern, observed_confusion AS correction_rule, frequency, 
           CASE severity WHEN 'CRITICAL' THEN 0.99 WHEN 'HIGH' THEN 0.95 ELSE 0.85 END AS confidence,
           updated_at
    FROM offline_confusion_patterns 
    ORDER BY frequency DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": r["id"],
            "pattern": r["pattern"],
            "correction_rule": r["correction_rule"],
            "frequency": r["frequency"],
            "confidence": r["confidence"],
            "updated_at": r["updated_at"]
        }
        for r in rows
    ]


def get_few_shot_prompt_context(utterance: str = "") -> str:
    """
    Deprecated / offline analysis only.
    Intentionally disabled in production extraction to prevent prompt injection
    and preserve uncorrupted extreme readings.
    """
    return ""


def generate_acoustic_confusion_report() -> Dict[str, Any]:
    """
    Generates an offline analytics report summarizing frequent ASR confusion pairs
    for human ML engineer review and model retraining datasets.
    """
    init_learning_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM learned_corrections WHERE review_status = 'PENDING_SUPERVISOR_AUDIT'")
    pending_count = cursor.fetchone()[0]

    cursor.execute("SELECT pattern, observed_confusion, frequency, severity FROM offline_confusion_patterns ORDER BY frequency DESC")
    patterns = [dict(r) for r in cursor.fetchall()]
    conn.close()

    return {
        "status": "OFFLINE_AUDIT_REPORT_READY",
        "pending_supervisor_reviews": pending_count,
        "documented_confusion_patterns": patterns,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def synthesize_new_rules() -> Dict[str, Any]:
    """Alias for backwards compatibility: generates offline acoustic confusion audit report."""
    return generate_acoustic_confusion_report()


def update_review_status(item_id: int, status: str, notes: Optional[str] = None) -> bool:
    """Supervisor marks an audit queue item as APPROVED, REJECTED, or RESOLVED."""
    return review_discrepancy(item_id=item_id, reviewer_id="FLEET_SUPERVISOR", decision=status, notes=notes)


def review_discrepancy(item_id: int, reviewer_id: str, decision: str, notes: Optional[str] = None) -> bool:
    """
    Supervisor marks an audit queue discrepancy with decision
    ('APPROVED_FOR_TRAINING', 'REJECTED', 'RESOLVED') and reviewer audit trail.
    """
    init_learning_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    full_notes = f"[Reviewer: {reviewer_id}] {notes or ''}".strip()
    cursor.execute("""
    UPDATE learned_corrections
    SET review_status = ?, reviewer_notes = ?
    WHERE id = ?
    """, (decision.upper(), full_notes, item_id))
    rows_affected = cursor.rowcount
    conn.commit()
    conn.close()
    return rows_affected > 0


def export_for_supervised_finetuning(output_path: Optional[str] = None, format: str = "jsonl") -> str:
    """
    Exports reviewed driver corrections as supervised training data (JSONL)
    for offline fine-tuning of Whisper or LLM extraction.
    Only exports entries where review_status = 'APPROVED_FOR_TRAINING'.
    """
    init_learning_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, vehicle_id, raw_transcript, original_extraction, driver_correction, corrected_slots
    FROM learned_corrections
    WHERE review_status = 'APPROVED_FOR_TRAINING'
    ORDER BY id ASC
    """)
    rows = cursor.fetchall()
    conn.close()

    import json
    out_file = Path(output_path) if output_path else Path(__file__).resolve().parent.parent / "data" / "supervised_corrections.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        for r in rows:
            entry = {
                "id": r["id"],
                "vehicle_id": r["vehicle_id"],
                "prompt": r["raw_transcript"],
                "driver_correction": r["driver_correction"],
                "target_extraction": json.loads(r["corrected_slots"]) if isinstance(r["corrected_slots"], str) else r["corrected_slots"]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Exported {len(rows)} approved samples to {out_file}")
    return str(out_file)
