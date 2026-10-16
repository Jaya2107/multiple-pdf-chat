
"""Shared fixtures: starts the Streamlit app once, creates a sample PDF."""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PORT = 8599
BASE_URL = f"http://localhost:{PORT}"


@pytest.fixture(scope="session")
def app_url():
    """Run `streamlit run app.py` in the background for the whole test session."""
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", str(PORT),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/_stcore/health", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(1)
    else:
        proc.terminate()
        raise RuntimeError("Streamlit app did not start in 60s")

    yield BASE_URL
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory):
    """A tiny PDF with a made-up fact, so we know the correct answer."""
    path = tmp_path_factory.mktemp("data") / "sample.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    lines = [
        "Zorland Country Fact Sheet",
        "The capital of Zorland is Quillon.",
        "The national currency of Zorland is the Zor.",
        "Zorland was founded in the year 1873 by Queen Marwa.",
        "The population of Zorland is approximately 4.2 million people.",
        "The official language of Zorland is Zorish, spoken by 90 percent of citizens.",
    ]
    y = 800
    for line in lines:
        c.drawString(72, y, line)
        y -= 28
    c.save()
    return path