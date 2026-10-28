"""Scheduled Phase 2 EOD command-line entrypoint."""

import json
import sys
import traceback
from pathlib import Path

from src.phase2.eod_automation import run_scheduled_eod


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        event = run_scheduled_eod(ROOT)
        print(json.dumps(event, ensure_ascii=False, indent=2))
        return 0

    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
