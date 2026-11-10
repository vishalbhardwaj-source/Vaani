"""
Vaani Voice Dispatcher Root Entrypoint
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Forward execution to app/dashboard.py
dashboard_path = PROJECT_ROOT / "app" / "dashboard.py"
with open(dashboard_path, "r", encoding="utf-8") as f:
    code = f.read()

exec(code, {"__name__": "__main__", "__file__": str(dashboard_path)})
