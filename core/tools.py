"""
Vaani: Tool Layer (Simulated Fleet Operations)
Interacts with data/fleet.sqlite to look up vehicles, trigger emergency assistance dispatches,
and record delay notices.

IMPORTANT: All outputs and logs from this module are explicitly marked 'Simulated fleet data'.
"""

import sqlite3
import datetime
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from core.schemas import VehicleRecord, AssistanceRequest, DispatchPriority

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "fleet.sqlite"
_db_initialized = False


def get_db_connection() -> sqlite3.Connection:
    global _db_initialized
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    if not _db_initialized:
        _db_initialized = True
        init_db()
    return conn


def init_db():
    """Ensure database schema is created and seeded with default simulated fleet."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS vehicles (
        vehicle_id TEXT PRIMARY KEY,
        driver_name TEXT NOT NULL,
        driver_phone TEXT NOT NULL,
        vehicle_type TEXT NOT NULL,
        cargo_type TEXT NOT NULL,
        target_temp_c REAL NOT NULL,
        current_temp_c REAL NOT NULL,
        route TEXT NOT NULL,
        last_known_location TEXT NOT NULL,
        status TEXT NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS assistance_requests (
        ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
        vehicle_id TEXT NOT NULL,
        incident_type TEXT NOT NULL,
        priority TEXT NOT NULL,
        location_reported TEXT NOT NULL,
        distance_km REAL,
        reported_temp_c REAL,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'DISPATCHED'
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS eta_updates (
        update_id INTEGER PRIMARY KEY AUTOINCREMENT,
        vehicle_id TEXT NOT NULL,
        destination TEXT NOT NULL,
        delay_minutes INTEGER NOT NULL,
        reason TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS supervisor_queue (
        escalation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        vehicle_id TEXT,
        driver_phone TEXT,
        trigger_reason TEXT NOT NULL,
        unconfirmed_temp_c REAL,
        telematics_sensor_c REAL,
        audio_recording_path TEXT,
        transcript_history TEXT,
        status TEXT DEFAULT 'PENDING',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        resolved_at TIMESTAMP,
        resolution_notes TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dispute_audit_log (
        audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticket_id INTEGER NOT NULL,
        vehicle_id TEXT NOT NULL,
        initial_audio_path TEXT,
        asr_transcript TEXT,
        readback_prompt_spoken TEXT,
        driver_reply_audio_path TEXT,
        driver_reply_intent TEXT,
        driver_reply_text TEXT,
        gate_decision TEXT,
        residual_risk_flag INTEGER DEFAULT 0,
        priority_source TEXT DEFAULT 'CONFIRMED_READING',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Migration checks for columns in existing DB
    cursor.execute("PRAGMA table_info(vehicles)")
    veh_cols = {row["name"] for row in cursor.fetchall()}
    if "gps_lat" not in veh_cols:
        cursor.execute("ALTER TABLE vehicles ADD COLUMN gps_lat REAL DEFAULT 28.6139")
    if "gps_lng" not in veh_cols:
        cursor.execute("ALTER TABLE vehicles ADD COLUMN gps_lng REAL DEFAULT 77.2090")
    if "gps_fix_age_minutes" not in veh_cols:
        cursor.execute("ALTER TABLE vehicles ADD COLUMN gps_fix_age_minutes INTEGER DEFAULT 4")
    if "speed_kmh" not in veh_cols:
        cursor.execute("ALTER TABLE vehicles ADD COLUMN speed_kmh REAL DEFAULT 60.0")

    cursor.execute("PRAGMA table_info(assistance_requests)")
    req_cols = {row["name"] for row in cursor.fetchall()}
    if "gps_lat" not in req_cols:
        cursor.execute("ALTER TABLE assistance_requests ADD COLUMN gps_lat REAL")
    if "gps_lng" not in req_cols:
        cursor.execute("ALTER TABLE assistance_requests ADD COLUMN gps_lng REAL")
    if "gps_fix_timestamp" not in req_cols:
        cursor.execute("ALTER TABLE assistance_requests ADD COLUMN gps_fix_timestamp TEXT")
    if "supervisor_alert_flag" not in req_cols:
        cursor.execute("ALTER TABLE assistance_requests ADD COLUMN supervisor_alert_flag INTEGER DEFAULT 0")
    if "residual_risk_flag" not in req_cols:
        cursor.execute("ALTER TABLE assistance_requests ADD COLUMN residual_risk_flag INTEGER DEFAULT 0")
    if "priority_source" not in req_cols:
        cursor.execute("ALTER TABLE assistance_requests ADD COLUMN priority_source TEXT DEFAULT 'CONFIRMED_READING'")

    cursor.execute("PRAGMA table_info(supervisor_queue)")
    sup_cols = {row["name"] for row in cursor.fetchall()}
    if "alert_level" not in sup_cols:
        cursor.execute("ALTER TABLE supervisor_queue ADD COLUMN alert_level TEXT DEFAULT 'STANDARD'")

    # Seed initial test data if empty
    cursor.execute("SELECT COUNT(*) FROM vehicles")
    if cursor.fetchone()[0] == 0:
        seed_data = [
            ("HR-55-TC-1102", "Ramesh Kumar", "+91-9876543210", "Reefer 32ft Multi-temp", "Ice Cream Consignment", -22.0, -14.0, "Karnal to Delhi", "NH-44 Corridor Milepost 118", "IN_TRANSIT", 29.6857, 76.9905, 4, 62.0),
            ("MH-12-RN-8819", "Sukhwinder Singh", "+91-9811223344", "Reefer 24ft Deep Freeze", "Vaccine Cold-Chain", -20.0, -18.5, "Pune to Mumbai", "Mumbai-Pune Expressway Section 4", "IN_TRANSIT", 18.7520, 73.4082, 5, 58.0),
            ("DL-01-AA-4092", "Abdul Sheikh", "+91-9899001122", "Reefer 20ft Chiller", "Dairy & Milk Products", 4.0, 9.5, "Jaipur to Delhi", "NH-48 Freight Corridor km 52", "CRITICAL_ALERT", 28.3241, 76.8872, 2, 54.0),
            ("KA-04-E-5541", "Manjunath Gowda", "+91-9744556677", "Reefer 32ft Heavy", "Frozen Meat Cargo", -18.0, -17.8, "Bangalore to Chennai", "Hosur Corridor Milepost 31", "IN_TRANSIT", 12.7409, 77.8253, 6, 60.0),
            ("GJ-01-BX-7201", "Vikram Patel", "+91-9822334455", "Reefer 16ft Quick-Serve", "Fresh Perishables / Fruits", 6.0, 14.2, "Ahmedabad to Surat", "Vadodara Logistics Hub Outer", "MAINTENANCE_REQUIRED", 22.3072, 73.1812, 3, 65.0)
        ]
        cursor.executemany("""
        INSERT INTO vehicles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, seed_data)

    conn.commit()
    conn.close()


def lookup_vehicle(vehicle_id: str) -> Optional[VehicleRecord]:
    """
    Look up vehicle record and telemetry.
    Returns: VehicleRecord or None
    Source: Simulated fleet data
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM vehicles WHERE vehicle_id = ?", (vehicle_id.strip(),))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    r_keys = row.keys()
    return VehicleRecord(
        vehicle_id=row["vehicle_id"],
        driver_name=row["driver_name"],
        driver_phone=row["driver_phone"],
        vehicle_type=row["vehicle_type"],
        cargo_type=row["cargo_type"],
        target_temp_c=row["target_temp_c"],
        current_temp_c=row["current_temp_c"],
        route=row["route"],
        last_known_location=row["last_known_location"],
        status=row["status"],
        gps_lat=row["gps_lat"] if "gps_lat" in r_keys else 28.6139,
        gps_lng=row["gps_lng"] if "gps_lng" in r_keys else 77.2090,
        gps_fix_age_minutes=row["gps_fix_age_minutes"] if "gps_fix_age_minutes" in r_keys else 4,
        speed_kmh=row["speed_kmh"] if "speed_kmh" in r_keys else 60.0
    )


def lookup_vehicle_by_phone(phone_number: str) -> Tuple[Optional[VehicleRecord], str]:
    """
    Caller-ID telematics lookup: identify vehicle from driver's mobile phone number.
    Extracts trailing 10 digits to match telephony formats (+91, leading 0, spaces).

    Safety Failure Modes Handled:
    1. Phone not in database -> returns (None, 'NOT_FOUND') -> forces prompt for registration.
    2. Phone shared across multiple trucks (or driver switched trucks) -> returns (None, 'MULTIPLE_MATCHES') -> forces prompt.
    3. Exactly one vehicle found -> returns (VehicleRecord, 'MATCH_FOUND').
    """
    clean_digits = re.sub(r"\D", "", str(phone_number))
    if len(clean_digits) >= 10:
        match_digits = clean_digits[-10:]
    else:
        return None, "INVALID_PHONE_FORMAT"

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM vehicles")
    rows = cursor.fetchall()
    conn.close()

    matching_vehicles = []
    for r in rows:
        stored_digits = re.sub(r"\D", "", str(r["driver_phone"]))
        if stored_digits.endswith(match_digits):
            r_keys = r.keys()
            veh = VehicleRecord(
                vehicle_id=r["vehicle_id"],
                driver_name=r["driver_name"],
                driver_phone=r["driver_phone"],
                vehicle_type=r["vehicle_type"],
                cargo_type=r["cargo_type"],
                target_temp_c=r["target_temp_c"],
                current_temp_c=r["current_temp_c"],
                route=r["route"],
                last_known_location=r["last_known_location"],
                status=r["status"],
                gps_lat=r["gps_lat"] if "gps_lat" in r_keys else 28.6139,
                gps_lng=r["gps_lng"] if "gps_lng" in r_keys else 77.2090,
                gps_fix_age_minutes=r["gps_fix_age_minutes"] if "gps_fix_age_minutes" in r_keys else 4,
                speed_kmh=r["speed_kmh"] if "speed_kmh" in r_keys else 60.0
            )
            matching_vehicles.append(veh)

    if len(matching_vehicles) == 0:
        return None, "NOT_FOUND"
    elif len(matching_vehicles) > 1:
        # Multiple trucks associated with the same phone (shared SIM or driver reassignment)
        # Never guess silently; must prompt driver for exact registration
        return None, "MULTIPLE_MATCHES"
    else:
        return matching_vehicles[0], "MATCH_FOUND"


def lookup_vehicle_by_registration_query(query: str) -> Tuple[Optional[VehicleRecord], str]:
    """
    Identifies a vehicle when caller-ID is unverified.
    Supports driver stating:
    - Last 4 digits of registration (e.g., '1102', 'gadi number 1102')
    - Full plate with or without hyphens/spaces (e.g., 'HR-55-TC-1102', 'HR 55 TC 1102', 'HR55TC1102')
    - Digits-only representation (e.g. '551102')

    Returns:
    - (VehicleRecord, 'MATCH_FOUND') if exactly one match.
    - (None, 'MULTIPLE_MATCHES') if query matches multiple vehicles.
    - (None, 'NOT_FOUND') if no vehicle matches.
    """
    raw = (query or "").strip()
    if not raw:
        return None, "NOT_FOUND"

    all_vehs = list_vehicles()
    if not all_vehs:
        return None, "NOT_FOUND"

    clean_alnum = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    digits_only = re.sub(r"\D", "", raw)

    matches = []
    for veh in all_vehs:
        veh_alnum = re.sub(r"[^A-Za-z0-9]", "", veh.vehicle_id).upper()
        veh_digits = re.sub(r"\D", "", veh.vehicle_id)

        # 1. Exact match on full registration string
        if veh.vehicle_id.upper() == raw.upper() or veh_alnum == clean_alnum:
            matches.append(veh)
            continue

        # 2. Match on 4-digit suffix (e.g. "1102")
        if len(digits_only) >= 4 and veh_digits.endswith(digits_only[-4:]):
            matches.append(veh)
            continue

        # 3. Match on full digits if at least 4 digits
        if len(digits_only) >= 4 and digits_only in veh_digits:
            matches.append(veh)
            continue

    # Deduplicate matches by vehicle_id
    unique_matches = {v.vehicle_id: v for v in matches}
    if len(unique_matches) == 1:
        return list(unique_matches.values())[0], "MATCH_FOUND"
    elif len(unique_matches) > 1:
        return None, "MULTIPLE_MATCHES"
    else:
        return None, "NOT_FOUND"


def list_vehicles() -> List[VehicleRecord]:
    """Return all vehicles in simulated fleet."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM vehicles ORDER BY vehicle_id ASC")
    rows = cursor.fetchall()
    conn.close()
    return [
        VehicleRecord(
            vehicle_id=r["vehicle_id"],
            driver_name=r["driver_name"],
            driver_phone=r["driver_phone"],
            vehicle_type=r["vehicle_type"],
            cargo_type=r["cargo_type"],
            target_temp_c=r["target_temp_c"],
            current_temp_c=r["current_temp_c"],
            route=r["route"],
            last_known_location=r["last_known_location"],
            status=r["status"],
            gps_lat=r["gps_lat"] if "gps_lat" in r.keys() else 28.6139,
            gps_lng=r["gps_lng"] if "gps_lng" in r.keys() else 77.2090,
            gps_fix_age_minutes=r["gps_fix_age_minutes"] if "gps_fix_age_minutes" in r.keys() else 4,
            speed_kmh=r["speed_kmh"] if "speed_kmh" in r.keys() else 60.0
        )
        for r in rows
    ]


def dispatch_assistance(
    vehicle_id: str,
    incident_type: str,
    location_reported: str,
    distance_km: Optional[float] = None,
    reported_temp_c: Optional[float] = None,
    priority: DispatchPriority = DispatchPriority.HIGH,
    notes: Optional[str] = None,
    supervisor_alert_flag: bool = False,
    residual_risk_flag: bool = False,
    gps_lat: Optional[float] = None,
    gps_lng: Optional[float] = None,
    priority_source: str = "CONFIRMED_READING"
) -> Dict[str, Any]:
    """
    Records a simulated roadside assistance and technician dispatch.
    Returns details with ticket ID and estimated technician arrival.
    Label: Simulated fleet data
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Determine priority based on cold-chain reefer thermal deviation or supervisor alert
    if supervisor_alert_flag:
        priority = DispatchPriority.CRITICAL
    elif reported_temp_c is not None and reported_temp_c > 10.0:
        priority = DispatchPriority.CRITICAL

    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    cursor.execute("""
    INSERT INTO assistance_requests (
        vehicle_id, incident_type, priority, location_reported, distance_km,
        reported_temp_c, notes, gps_lat, gps_lng, gps_fix_timestamp,
        supervisor_alert_flag, residual_risk_flag, priority_source
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        vehicle_id,
        incident_type,
        priority.value,
        location_reported,
        distance_km,
        reported_temp_c,
        f"[Simulated fleet data] {notes or 'Spoken dispatch triggered via Vaani'} [Priority source: {priority_source}]",
        gps_lat,
        gps_lng,
        now_iso,
        1 if supervisor_alert_flag else 0,
        1 if residual_risk_flag else 0,
        priority_source
    ))

    ticket_id = cursor.lastrowid

    # Update vehicle status in simulated fleet
    cursor.execute("""
    UPDATE vehicles 
    SET status = 'ASSISTANCE_DISPATCHED'
    WHERE vehicle_id = ?
    """, (vehicle_id,))

    conn.commit()
    conn.close()

    eta_minutes = 25 if priority == DispatchPriority.CRITICAL else 45

    return {
        "status": "DISPATCHED",
        "ticket_id": ticket_id,
        "vehicle_id": vehicle_id,
        "incident_type": incident_type,
        "priority": priority.value,
        "location": f"{location_reported} (km {distance_km})" if distance_km else location_reported,
        "reported_temp_c": reported_temp_c,
        "technician_eta_min": eta_minutes,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "Simulated fleet data"
    }


def update_eta(
    vehicle_id: str,
    destination: str,
    delay_minutes: int,
    reason: str
) -> Dict[str, Any]:
    """
    Records a simulated delay notice to destination depot / customer.
    Label: Simulated fleet data
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO eta_updates (vehicle_id, destination, delay_minutes, reason)
    VALUES (?, ?, ?, ?)
    """, (
        vehicle_id,
        destination,
        delay_minutes,
        f"[Simulated fleet data] {reason}"
    ))
    update_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return {
        "status": "ETA_UPDATED",
        "update_id": update_id,
        "vehicle_id": vehicle_id,
        "destination": destination,
        "delay_minutes": delay_minutes,
        "source": "Simulated fleet data"
    }


def get_action_log(limit: int = 15) -> List[Dict[str, Any]]:
    """Fetch recent simulated dispatches from SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT ticket_id, vehicle_id, incident_type, priority, location_reported, 
           distance_km, reported_temp_c, notes, created_at, status
    FROM assistance_requests 
    ORDER BY ticket_id DESC 
    LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "ticket_id": f"TKT-{r['ticket_id']:04d}",
            "vehicle_id": r["vehicle_id"],
            "incident_type": r["incident_type"],
            "priority": r["priority"],
            "location": f"{r['location_reported']} (+{r['distance_km']}km)" if r["distance_km"] else r["location_reported"],
            "temp_c": f"{r['reported_temp_c']}°C" if r["reported_temp_c"] is not None else "N/A",
            "created_at": r["created_at"],
            "status": r["status"],
            "source": "Simulated fleet data"
        }
        for r in rows
    ]


def enqueue_supervisor_escalation(
    vehicle_id: Optional[str],
    trigger_reason: str,
    driver_phone: Optional[str] = None,
    unconfirmed_temp_c: Optional[float] = None,
    telematics_sensor_c: Optional[float] = None,
    audio_recording_path: Optional[str] = None,
    transcript_history: Optional[str] = None,
    alert_level: str = "STANDARD"
) -> int:
    """
    Pushes an unresolved or safety-escalated call into the real supervisor dispatch queue.
    Supports alert_level='UNVERIFIED_SENSOR_CORROBORATED' when sensor corroborates an unconfirmed excursion.
    Returns: escalation_id
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Deduplicate pending tickets per vehicle to prevent redial flooding:
    # Keeps highest priority, appends call event history, and respects a 120-minute time window.
    if vehicle_id:
        cursor.execute("""
        SELECT escalation_id, alert_level, transcript_history, created_at FROM supervisor_queue 
        WHERE vehicle_id = ? AND status = 'PENDING'
        ORDER BY escalation_id DESC LIMIT 1
        """, (vehicle_id,))
        existing = cursor.fetchone()
        if existing:
            created_str = existing["created_at"]
            is_recent = True
            if created_str:
                try:
                    # SQLite default created_at format is "YYYY-MM-DD HH:MM:SS"
                    created_dt = datetime.datetime.strptime(created_str.split(".")[0], "%Y-%m-%d %H:%M:%S")
                    age_minutes = (datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) - created_dt).total_seconds() / 60.0
                    if age_minutes > 120:
                        is_recent = False
                except Exception:
                    pass

            if is_recent:
                esc_id = existing["escalation_id"]
                # Keep highest priority: if either is UNVERIFIED_SENSOR_CORROBORATED, preserve it!
                effective_alert = (
                    "UNVERIFIED_SENSOR_CORROBORATED" 
                    if ("UNVERIFIED_SENSOR_CORROBORATED" in (alert_level, existing["alert_level"])) 
                    else alert_level
                )
                # Append each call as a timestamped event rather than replacing history
                old_hist = existing["transcript_history"] or ""
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                appended_hist = f"{old_hist}\n[REDIAL {now_str}]: {transcript_history}" if old_hist else transcript_history

                cursor.execute("""
                UPDATE supervisor_queue
                SET trigger_reason = ?,
                    driver_phone = COALESCE(?, driver_phone),
                    unconfirmed_temp_c = COALESCE(?, unconfirmed_temp_c),
                    telematics_sensor_c = COALESCE(?, telematics_sensor_c),
                    audio_recording_path = COALESCE(?, audio_recording_path),
                    transcript_history = ?, alert_level = ?
                WHERE escalation_id = ?
                """, (
                    trigger_reason,
                    driver_phone,
                    unconfirmed_temp_c,
                    telematics_sensor_c,
                    audio_recording_path,
                    appended_hist,
                    effective_alert,
                    esc_id
                ))
                conn.commit()
                conn.close()
                return esc_id

    cursor.execute("""
    INSERT INTO supervisor_queue (
        vehicle_id, driver_phone, trigger_reason, unconfirmed_temp_c,
        telematics_sensor_c, audio_recording_path, transcript_history, status, alert_level
    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
    """, (
        vehicle_id,
        driver_phone,
        trigger_reason,
        unconfirmed_temp_c,
        telematics_sensor_c,
        audio_recording_path,
        transcript_history,
        alert_level
    ))
    esc_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return esc_id


def list_supervisor_queue(status: Optional[str] = "PENDING") -> List[Dict[str, Any]]:
    """Fetches tickets from the live supervisor escalation queue."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if status:
        cursor.execute("SELECT * FROM supervisor_queue WHERE status = ? ORDER BY escalation_id DESC", (status,))
    else:
        cursor.execute("SELECT * FROM supervisor_queue ORDER BY escalation_id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def resolve_supervisor_escalation(escalation_id: int, resolution_notes: str) -> bool:
    """Marks an escalation ticket as resolved by human fleet supervisor."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    cursor.execute("""
    UPDATE supervisor_queue 
    SET status = 'RESOLVED', resolved_at = ?, resolution_notes = ?
    WHERE escalation_id = ?
    """, (now_iso, resolution_notes, escalation_id))
    rows_affected = cursor.rowcount
    conn.commit()
    conn.close()
    return rows_affected > 0


def record_dispute_audit(
    ticket_id: int,
    vehicle_id: str,
    asr_transcript: str,
    readback_prompt_spoken: str,
    driver_reply_intent: str,
    driver_reply_text: Optional[str] = None,
    initial_audio_path: Optional[str] = None,
    driver_reply_audio_path: Optional[str] = None,
    gate_decision: str = "ALLOW",
    residual_risk_flag: bool = False,
    priority_source: str = "CONFIRMED_READING"
) -> int:
    """
    Stores an immutable forensic audit bundle for every dispatch.
    Links raw audio path, transcript, spoken read-back, and driver reply.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO dispute_audit_log (
        ticket_id, vehicle_id, initial_audio_path, asr_transcript,
        readback_prompt_spoken, driver_reply_audio_path, driver_reply_intent,
        driver_reply_text, gate_decision, residual_risk_flag, priority_source
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ticket_id,
        vehicle_id,
        initial_audio_path,
        asr_transcript,
        readback_prompt_spoken,
        driver_reply_audio_path,
        driver_reply_intent,
        driver_reply_text,
        gate_decision,
        1 if residual_risk_flag else 0,
        priority_source
    ))
    audit_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return audit_id


def get_dispute_audit(ticket_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves full forensic dispute bundle by dispatch ticket ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM dispute_audit_log WHERE ticket_id = ? ORDER BY audit_id DESC LIMIT 1", (ticket_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

