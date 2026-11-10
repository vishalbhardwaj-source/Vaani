import os
import re
from pathlib import Path
import whisper

p = Path(r"C:\Users\visha\Downloads\VoiceAI\real voice")
model = whisper.load_model("base")

script_sentences = [
    (1, "Bhaiya, reefer display minus 18 degree dikha raha hai, Panipat se 25 km pehle."),
    (2, "Temperature 18 degree ho gaya hai, toll plaza se 42 km aage."),
    (3, "Cabin ka temperature 14 degree hai, Manesar se 35 km door."),
    (4, "Sir, minus 14 degree ho gaya, Hosur se 12 km aage."),
    (5, "Reefer 8 degree dikha raha hai, 80 km pehle breakdown hua."),
    (6, "Minus 8 degree hai, Kundli Border se 15 km aage."),
    (7, "Chiller temperature 5 degree hai, Talegaon se 18 km door."),
    (8, "Temperature minus 20 degree, Dharuhera se 40 km pehle."),
    (9, "Freezer 4 degree par chal raha hai, 14 km aage rukna hai."),
    (10, "Reefer minus 2 degree dikha raha hai, Mumbai Pune Expressway 68 km par."),
    (11, "The temperature is 18 degrees and I am 80 kilometres from the depot."),
    (12, "Temperature is minus 18 degrees, I'm 14 kilometres from the toll."),
]

def parse_time(f):
    m = re.search(r'(\d{4}-\d{2}-\d{2}) at (\d+)\.(\d+)\.(\d+) (AM|PM)(?: \((\d+)\))?', f.name)
    if not m:
        return (0, 0, f.name)
    dstr, hh, mm, ss, ampm, dup = m.groups()
    hh = int(hh)
    if ampm == 'PM' and hh != 12: hh += 12
    if ampm == 'AM' and hh == 12: hh = 0
    t = f"{dstr} {hh:02d}:{mm}:{ss}"
    dup_val = int(dup) if dup else 0
    return (t, dup_val, f.name)

# Group files
sessions = {
    "Speaker 1 (2026-11-08 Night)": [],
    "Speaker 2 (2026-11-09 Afternoon 1:48 PM)": [],
    "Speaker 3 (2026-11-09 Afternoon 3:15 PM)": [],
    "Speaker 4 (2026-11-09 Night 11:41 PM)": [],
}

for f in p.glob("*.ogg"):
    n = f.name
    if "2026-11-08 at 11.1" in n:
        sessions["Speaker 1 (2026-11-08 Night)"].append(f)
    elif "2026-11-09 at 1.4" in n or "2026-11-09 at 1.5" in n:
        sessions["Speaker 2 (2026-11-09 Afternoon 1:48 PM)"].append(f)
    elif "2026-11-09 at 3.1" in n:
        sessions["Speaker 3 (2026-11-09 Afternoon 3:15 PM)"].append(f)
    elif "Audio 2026-11-09 at 11.41" in n:
        sessions["Speaker 4 (2026-11-09 Night 11:41 PM)"].append(f)

print("="*60)
for s_name, flist in sessions.items():
    s_files = sorted(flist, key=parse_time)
    print(f"\n{s_name}: {len(s_files)} files")
    for idx, f in enumerate(s_files):
        # avoid duplicate re-run if byte identical
        print(f"  [{idx+1}] {f.name} ({f.stat().st_size} bytes)")
