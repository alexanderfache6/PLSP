#!/usr/bin/env python3
"""Run the whole data availability pipeline, in order.

Refreshes the 3DEP, NAIP and NEON availability CSVs, then redraws the timeline
from them.
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = [
    ("3DEP", "run_3dep_availability.py"),
    ("NAIP", "run_naip_availability.py"),
    ("NEON", "run_neon_availability.py"),
    ("timeline", "run_create_availability_timeline.py"),
]


def main():
    for label, script in STEPS:
        print(f"\n{'=' * 70}\n== {label}: {script}\n{'=' * 70}", flush=True)
        started = time.monotonic()
        subprocess.run([sys.executable, str(HERE / script)])
        print(f"-- {label}: {time.monotonic() - started:.1f}s", flush=True)

    print(f"\n{'=' * 70}\nAll {len(STEPS)} steps completed.")


if __name__ == "__main__":
    main()
