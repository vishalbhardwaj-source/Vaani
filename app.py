"""
Vaani Voice Dispatcher Root Entrypoint
"""
import sys
from pathlib import Path

# Forward execution to app/dashboard.py
dashboard_path = Path(__file__).resolve().parent / "app" / "dashboard.py"
with open(dashboard_path, "r", encoding="utf-8") as f:
    code = f.read()

exec(code, {"__name__": "__main__", "__file__": str(dashboard_path)})
