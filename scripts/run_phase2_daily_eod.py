"""Run Phase 2 EOD data ingestion."""

import json
from pathlib import Path

from src.phase2.daily_data import run_daily


ROOT = Path(__file__).resolve().parents[1]


def main():
    result = run_daily(ROOT)

    print("\nPHASE 2 DAILY DATA REPORT")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()