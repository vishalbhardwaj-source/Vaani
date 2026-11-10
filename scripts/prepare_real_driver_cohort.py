import os
import re
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_VOICE_DIR = PROJECT_ROOT / "real voice"
OUT_DIR = PROJECT_ROOT / "data" / "real_driver_cohort"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 12 Ground truth sentences
SCRIPT_SENTENCES = [
    {
        "sentence_id": 1,
        "text": "Bhaiya, reefer display minus 18 degree dikha raha hai, Panipat se 25 km pehle.",
        "true_temp": -18.0,
        "true_dist": 25.0,
        "true_loc": "Panipat",
        "incident_type": "reefer_temperature",
        "cargo": "Ice Cream / Deep Freeze",
        "vehicle_id": "HR-55-TC-1102"
    },
    {
        "sentence_id": 2,
        "text": "Temperature 18 degree ho gaya hai, toll plaza se 42 km aage.",
        "true_temp": 18.0,
        "true_dist": 42.0,
        "true_loc": "Toll Plaza",
        "incident_type": "reefer_temperature",
        "cargo": "Produce / Fresh Goods",
        "vehicle_id": "GJ-01-BX-7201"
    },
    {
        "sentence_id": 3,
        "text": "Cabin ka temperature 14 degree hai, Manesar se 35 km door.",
        "true_temp": 14.0,
        "true_dist": 35.0,
        "true_loc": "Manesar",
        "incident_type": "reefer_temperature",
        "cargo": "Chilled Dairy",
        "vehicle_id": "DL-01-AA-4092"
    },
    {
        "sentence_id": 4,
        "text": "Sir, minus 14 degree ho gaya, Hosur se 12 km aage.",
        "true_temp": -14.0,
        "true_dist": 12.0,
        "true_loc": "Hosur",
        "incident_type": "reefer_temperature",
        "cargo": "Chilled Meat",
        "vehicle_id": "KA-04-E-5541"
    },
    {
        "sentence_id": 5,
        "text": "Reefer 8 degree dikha raha hai, 80 km pehle breakdown hua.",
        "true_temp": 8.0,
        "true_dist": 80.0,
        "true_loc": None,
        "incident_type": "breakdown",
        "cargo": "Dairy Tanker",
        "vehicle_id": "PB-10-CZ-4410"
    },
    {
        "sentence_id": 6,
        "text": "Minus 8 degree hai, Kundli Border se 15 km aage.",
        "true_temp": -8.0,
        "true_dist": 15.0,
        "true_loc": "Kundli Border",
        "incident_type": "reefer_temperature",
        "cargo": "Frozen Seafood",
        "vehicle_id": "HR-26-BR-9912"
    },
    {
        "sentence_id": 7,
        "text": "Chiller temperature 5 degree hai, Talegaon se 18 km door.",
        "true_temp": 5.0,
        "true_dist": 18.0,
        "true_loc": "Talegaon",
        "incident_type": "reefer_temperature",
        "cargo": "Dairy Products",
        "vehicle_id": "MH-04-DK-3388"
    },
    {
        "sentence_id": 8,
        "text": "Temperature minus 20 degree, Dharuhera se 40 km pehle.",
        "true_temp": -20.0,
        "true_dist": 40.0,
        "true_loc": "Dharuhera",
        "incident_type": "reefer_temperature",
        "cargo": "Vaccine Cold-Chain",
        "vehicle_id": "MH-12-RN-8819"
    },
    {
        "sentence_id": 9,
        "text": "Freezer 4 degree par chal raha hai, 14 km aage rukna hai.",
        "true_temp": 4.0,
        "true_dist": 14.0,
        "true_loc": None,
        "incident_type": "reefer_temperature",
        "cargo": "Dairy Consignment",
        "vehicle_id": "DL-01-AA-4092"
    },
    {
        "sentence_id": 10,
        "text": "Reefer minus 2 degree dikha raha hai, Mumbai Pune Expressway 68 km par.",
        "true_temp": -2.0,
        "true_dist": 68.0,
        "true_loc": "Mumbai Pune Expressway",
        "incident_type": "reefer_temperature",
        "cargo": "Vaccine Cold-Chain",
        "vehicle_id": "MH-12-RN-8819"
    },
    {
        "sentence_id": 11,
        "text": "The temperature is 18 degrees and I am 80 kilometres from the depot.",
        "true_temp": 18.0,
        "true_dist": 80.0,
        "true_loc": "Depot",
        "incident_type": "reefer_temperature",
        "cargo": "General Reefer",
        "vehicle_id": "GJ-01-BX-7201"
    },
    {
        "sentence_id": 12,
        "text": "Temperature is minus 18 degrees, I'm 14 kilometres from the toll.",
        "true_temp": -18.0,
        "true_dist": 14.0,
        "true_loc": "Toll",
        "incident_type": "reefer_temperature",
        "cargo": "Deep Freeze Ice Cream",
        "vehicle_id": "HR-55-TC-1102"
    }
]

# Let's map the files for each speaker
# Speaker 1 files (sorted chronologically):
spk1_files = [
    "WhatsApp Ptt 2026-11-08 at 11.17.58 PM.ogg", # 1
    "WhatsApp Ptt 2026-11-08 at 11.18.05 PM.ogg", # 2
    "WhatsApp Ptt 2026-11-08 at 11.18.11 PM.ogg", # 3
    "WhatsApp Ptt 2026-11-08 at 11.18.30 PM.ogg", # 4
    "WhatsApp Ptt 2026-11-08 at 11.18.36 PM.ogg", # 5
    "WhatsApp Ptt 2026-11-08 at 11.18.41 PM.ogg", # 6
    "WhatsApp Ptt 2026-11-08 at 11.18.48 PM.ogg", # 7
    "WhatsApp Ptt 2026-11-08 at 11.18.56 PM.ogg", # 8
    "WhatsApp Ptt 2026-11-08 at 11.19.06 PM.ogg", # 9
    "WhatsApp Ptt 2026-11-08 at 11.19.14 PM.ogg", # 10
    "WhatsApp Ptt 2026-11-08 at 11.19.20 PM.ogg", # 11
    "WhatsApp Ptt 2026-11-08 at 11.19.39 PM.ogg", # 12
]

# Speaker 2 files:
spk2_files = [
    "WhatsApp Ptt 2026-11-09 at 1.48.07 PM.ogg", # 1
    "WhatsApp Ptt 2026-11-09 at 1.48.22 PM.ogg", # 2
    "WhatsApp Ptt 2026-11-09 at 1.48.33 PM.ogg", # 3
    "WhatsApp Ptt 2026-11-09 at 1.48.45 PM.ogg", # 4
    "WhatsApp Ptt 2026-11-09 at 1.49.00 PM.ogg", # 5
    "WhatsApp Ptt 2026-11-09 at 1.49.48 PM.ogg", # 6
    "WhatsApp Ptt 2026-11-09 at 1.50.29 PM.ogg", # 7
    "WhatsApp Ptt 2026-11-09 at 1.50.49 PM.ogg", # 8
    "WhatsApp Ptt 2026-11-09 at 1.50.58 PM.ogg", # 9
    "WhatsApp Ptt 2026-11-09 at 1.51.09 PM.ogg", # 10
    "WhatsApp Ptt 2026-11-09 at 1.51.21 PM.ogg", # 11
    "WhatsApp Ptt 2026-11-09 at 1.51.41 PM.ogg", # 12
]

# Speaker 3 files:
# Let's inspect which files in session 3 map to 1..12
spk3_raw = [
    "WhatsApp Ptt 2026-11-09 at 3.15.33 PM.ogg",
    "WhatsApp Ptt 2026-11-09 at 3.15.42 PM.ogg",
    "WhatsApp Ptt 2026-11-09 at 3.15.52 PM.ogg",
    "WhatsApp Ptt 2026-11-09 at 3.16.08 PM.ogg",
    "WhatsApp Ptt 2026-11-09 at 3.16.17 PM.ogg",
    "WhatsApp Ptt 2026-11-09 at 3.16.30 PM.ogg",
    "WhatsApp Ptt 2026-11-09 at 3.16.46 PM.ogg",
    "WhatsApp Ptt 2026-11-09 at 3.16.55 PM.ogg",
    "WhatsApp Ptt 2026-11-09 at 3.17.08 PM.ogg",
    "WhatsApp Ptt 2026-11-09 at 3.17.18 PM.ogg",
    "WhatsApp Ptt 2026-11-09 at 3.17.31 PM.ogg",
    "WhatsApp Ptt 2026-11-09 at 11.05.01 AM.ogg" # or standalone
]

# Speaker 4 files:
spk4_raw = [
    "WhatsApp Audio 2026-11-09 at 11.41.21 PM.ogg",
    "WhatsApp Audio 2026-11-09 at 11.41.21 PM (1).ogg",
    "WhatsApp Audio 2026-11-09 at 11.41.22 PM.ogg",
    "WhatsApp Audio 2026-11-09 at 11.41.22 PM (1).ogg",
    "WhatsApp Audio 2026-11-09 at 11.41.22 PM (2).ogg",
    "WhatsApp Audio 2026-11-09 at 11.41.23 PM.ogg",
    "WhatsApp Audio 2026-11-09 at 11.41.23 PM (1).ogg",
    "WhatsApp Audio 2026-11-09 at 11.41.23 PM (2).ogg",
    "WhatsApp Audio 2026-11-09 at 11.41.23 PM (3).ogg",
    "WhatsApp Audio 2026-11-09 at 11.41.24 PM.ogg",
    "WhatsApp Audio 2026-11-09 at 11.41.24 PM (1).ogg",
    "WhatsApp Audio 2026-11-09 at 11.41.24 PM (2).ogg",
]

print("Ground truth sentences count:", len(SCRIPT_SENTENCES))
