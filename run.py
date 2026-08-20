"""
MP Urban Traffic Forecasting System — 1-Click Launcher
Runs frontend build check, launches FastAPI server, and opens default web browser.
"""

import os
import sys
import time
import subprocess
import webbrowser
import threading
from pathlib import Path

# Ensure UTF-8 output encoding if possible
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent

def build_frontend_if_needed():
    """Builds the React frontend SPA if dist directory doesn't exist."""
    dist_index = PROJECT_ROOT / "frontend" / "dist" / "index.html"
    if not dist_index.exists():
        print("[1/2] Building React frontend bundle...")
        npm_cmd = "cmd /c npm --prefix frontend run build" if sys.platform == "win32" else "npm --prefix frontend run build"
        result = subprocess.run(npm_cmd, shell=True)
        if result.returncode != 0:
            print("[WARNING] Frontend build returned an error. Using existing assets if available.")
        else:
            print("[OK] Frontend bundle built successfully.")
    else:
        print("[1/2] Production frontend assets ready.")

def open_browser():
    """Opens browser after server startup delay."""
    time.sleep(2)
    print("\n[INFO] Opening application in your default browser at http://localhost:7860 ...\n")
    webbrowser.open("http://localhost:7860")

def main():
    print("=" * 70)
    print("  MP Urban Traffic Congestion Forecasting System")
    print("  Starting system on http://127.0.0.1:7860")
    print("=" * 70)

    # 1. Build frontend if needed
    build_frontend_if_needed()

    # 2. Open browser automatically in a background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # 3. Start Uvicorn FastAPI application
    print("[2/2] Starting FastAPI backend server...")
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=7860, reload=False)

if __name__ == "__main__":
    main()
