"""Documented command-line demonstrations should remain runnable."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_environment_demo_runs_with_current_action_and_info_schema():
    result = subprocess.run(
        [sys.executable, "-B", "env.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "OBSERVATION BREAKDOWN" in result.stdout
