"""
Saarthi: Tool Layer (Simulated Fleet Operations)
Interacts with data/fleet.sqlite to look up vehicles, trigger emergency assistance dispatches,
and record delay notices.

IMPORTANT: All outputs and logs from this module are explicitly marked 'Simulated fleet data'.
"""

import sqlite3
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from core.schemas import VehicleRecord, AssistanceRequest, DispatchPriority

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "fleet.sqlite"


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Ensure database schema is created and seeded with default simulated fleet."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_db_connection()
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

    # Seed initial test data if empty
    cursor.execute("SELECT COUNT(*) FROM vehicles")
    if cursor.fetchone()[0] == 0:
        seed_data = [
            ("HR-55-TC-1102", "Ramesh Kumar", "+91-9876543210", "Reefer 32ft Multi-temp", "Ice Cream Consignment", -22.0, -14.0, "Karnal to Delhi", "NH-44 Toll Plaza km 42", "IN_TRANSIT"),
            ("MH-12-RN-8819", "Sukhwinder Singh", "+91-9811223344", "Reefer 24ft Deep Freeze", "Vaccine Cold-Chain", -20.0, -18.5, "Pune to Mumbai", "Mumbai-Pune Expressway km 68", "IN_TRANSIT"),
            ("DL-01-AA-4092", "Abdul Sheikh", "+91-9899001122", "Reefer 20ft Chiller", "Dairy & Milk Products", 4.0, 9.5, "Jaipur to Delhi", "NH-48 Bilaspur Chowk", "CRITICAL_ALERT"),
            ("KA-04-E-5541", "Manjunath Gowda", "+91-9744556677", "Reefer 32ft Heavy", "Frozen Meat Cargo", -18.0, -17.8, "Bangalore to Chennai", "Hosur Highway km 35", "IN_TRANSIT"),
            ("GJ-01-BX-7201", "Vikram Patel", "+91-9822334455", "Reefer 16ft Quick-Serve", "Fresh Perishables / Fruits", 6.0, 14.2, "Ahmedabad to Surat", "Vadodara Bypass km 80", "MAINTENANCE_REQUIRED")
        ]
        cursor.executemany("""
        INSERT INTO vehicles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        status=row["status"]
    )


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
            status=r["status"]
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
    notes: Optional[str] = None
) -> Dict[str, Any]:
    """
    Records a simulated roadside assistance and technician dispatch.
    Returns details with ticket ID and estimated technician arrival.
    Label: Simulated fleet data
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Determine priority based on cold-chain reefer thermal deviation
    if reported_temp_c is not None and reported_temp_c > 10.0:
        priority = DispatchPriority.CRITICAL

    cursor.execute("""
    INSERT INTO assistance_requests (
        vehicle_id, incident_type, priority, location_reported, distance_km, reported_temp_c, notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        vehicle_id,
        incident_type,
        priority.value,
        location_reported,
        distance_km,
        reported_temp_c,
        f"[Simulated fleet data] {notes or 'Spoken dispatch triggered via Saarthi'}"
    ))

    ticket_id = cursor.lastrowid

    # Update vehicle status in simulated fleet
    cursor.execute("""
    UPDATE vehicles 
    SET status = 'ASSISTANCE_DISPATCHED', current_temp_c = COALESCE(?, current_temp_c)
    WHERE vehicle_id = ?
    """, (reported_temp_c, vehicle_id))

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
