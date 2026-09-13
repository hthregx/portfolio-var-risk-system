from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    "scripts/generate_ewma_comparison.py",
    "scripts/generate_ewma_figures.py",
    "scripts/audit_ewma_comparison.py",
    "scripts/generate_ewma_metadata.py",
]


def run_step(script: str) -> None:
    path = PROJECT_ROOT / script

    if not path.exists():
        raise FileNotFoundError(
            f"Required pipeline step not found: {script}"
        )

    print(f"\n=== Running {script} ===")

    subprocess.run(
        [sys.executable, str(path)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    print("Portfolio VaR baseline pipeline")
    print(f"Project root: {PROJECT_ROOT}")

    for script in STEPS:
        run_step(script)

    print("\nB-14.4 RUN PIPELINE PASS")


if __name__ == "__main__":
    main()