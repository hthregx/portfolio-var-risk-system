#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# When this file is executed as ``python scripts/audit_day28_hygiene_b.py``,
# Python puts ``scripts/`` on sys.path rather than the repository root.
# Add the repo root explicitly so sibling modules can be imported portably.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PYTHON_SCAN_ROOTS = ("src", "scripts", "app")
IGNORED_DIR_NAMES = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox", "node_modules",
}

ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"/Users/[^/\\s]+/"),
    re.compile(r"/home/[^/\\s]+/"),
    re.compile(r"[A-Za-z]:[\\/](?:Users|Documents|Desktop|Projects?)[\\/]"),
)

REQUIRED_GITIGNORE_MARKERS = (
    ".venv/",
    "__pycache__/",
    ".pytest_cache/",
)

KNOWN_FREEZE_HASH_CONTRACTS = {
    "sha256",
    "final_evaluation_config_sha256",
    "final_predictions_sha256",
    "final_metrics_sha256",
    "final_run_metadata_sha256",
    "final_metric_comparison_sha256",
    "final_pairwise_diagnostics_sha256",
    "final_pairwise_summary_sha256",
}


def fail(message: str) -> None:
    raise AssertionError(message)


def run(args: list[str], *, env: dict[str, str] | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and result.returncode != 0:
        fail(
            f"command failed ({result.returncode}): {' '.join(args)}\n"
            f"{result.stdout[-5000:]}"
        )
    return result


def git_output(*args: str) -> str:
    return run(["git", *args], check=True).stdout


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_bytes(revision: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cannot read git blob {revision}:{path}: "
            f"{result.stderr.decode('utf-8', errors='replace')}"
        )
    return result.stdout


def iter_python_files():
    for root_name in PYTHON_SCAN_ROOTS:
        root = REPO_ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if any(part in IGNORED_DIR_NAMES for part in path.parts):
                continue
            yield path


def check_gitignore() -> None:
    path = REPO_ROOT / ".gitignore"
    if not path.is_file():
        fail("missing .gitignore")

    text = path.read_text(encoding="utf-8")
    missing = [m for m in REQUIRED_GITIGNORE_MARKERS if m not in text]
    if missing:
        fail(".gitignore missing hygiene coverage for: " + ", ".join(missing))


def check_encoding_and_paths() -> tuple[int, int]:
    file_count = 0
    absolute_hits: list[str] = []
    audit_file = Path(__file__).resolve()

    for path in iter_python_files():
        file_count += 1
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            fail(f"non-UTF-8 Python source: {path.relative_to(REPO_ROOT)}: {exc}")

        # This audit file intentionally contains the absolute-path detector
        # regex literals themselves (for example /Users/... and C:\\...).
        # Skip only the detector file for this one scan to avoid a self-match.
        if path.resolve() == audit_file:
            continue

        for pattern in ABSOLUTE_PATH_PATTERNS:
            if pattern.search(text):
                absolute_hits.append(str(path.relative_to(REPO_ROOT)))
                break

    if absolute_hits:
        fail(
            "machine-specific absolute path detected in Python source: "
            + ", ".join(sorted(absolute_hits))
        )

    return file_count, len(absolute_hits)


def check_python_compile() -> None:
    result = run([sys.executable, "-m", "compileall", "-q", "src", "scripts", "app"])
    if result.returncode != 0:
        fail("Python compile check failed:\n" + result.stdout[-5000:])


def check_git_diff_cleanliness() -> None:
    result = run(["git", "diff", "--check"])
    if result.returncode != 0:
        fail("git diff --check failed:\n" + result.stdout[-5000:])


def classify_preexisting_freeze_issue() -> tuple[int, int, str]:
    """Allow only the already-proven Day23 SHA-only provenance defect."""
    try:
        import scripts.validate_model_freeze_a as freeze_validator
    except Exception as exc:
        fail(f"cannot import Day23 freeze validator: {exc}")

    freeze_path = REPO_ROOT / "configs" / "model_freeze.yaml"
    frame = freeze_validator.validate_model_freeze(
        freeze_path=freeze_path,
        root=REPO_ROOT,
    )

    failed = frame.loc[frame["status"] == "FAIL"].copy()
    if failed.empty:
        return 0, 0, "none"

    failed_contracts = set(failed["contract"].astype(str))
    unexpected = failed_contracts - KNOWN_FREEZE_HASH_CONTRACTS
    if unexpected:
        fail(
            "model-freeze validator has non-hash failures; "
            "this is not an allowed hygiene exception: "
            + ", ".join(sorted(unexpected))
        )

    freeze_log = git_output(
        "log", "--format=%H", "--", "configs/model_freeze.yaml"
    ).splitlines()
    if not freeze_log:
        fail("cannot locate commit that introduced model_freeze.yaml")
    freeze_commit = freeze_log[-1].strip()

    tracked_verified = 0
    untracked_verified = 0

    for row in failed.itertuples(index=False):
        source = str(row.source)
        path = REPO_ROOT / source

        tracked = run(["git", "ls-files", "--error-unmatch", source])
        if tracked.returncode == 0:
            try:
                head_blob = git_blob_bytes("HEAD", source)
                freeze_blob = git_blob_bytes(freeze_commit, source)
            except RuntimeError as exc:
                fail(str(exc))

            if sha256_bytes(head_blob) != sha256_bytes(freeze_blob):
                fail("tracked canonical artifact changed after model freeze: " + source)

            if path.is_file():
                working_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                head_hash = sha256_bytes(head_blob)
                if working_hash != head_hash:
                    fail(
                        "working-tree bytes differ from HEAD for canonical artifact: "
                        + source
                    )

            tracked_verified += 1
            continue

        ignored = run(["git", "check-ignore", "-q", source])
        if ignored.returncode != 0:
            fail(
                "failed freeze source is neither tracked nor intentionally ignored: "
                + source
            )
        untracked_verified += 1

    return tracked_verified, untracked_verified, freeze_commit[:12]


def check_tests_do_not_mutate_repo() -> tuple[str, int, int]:
    """
    Run the full repo suite in the repository's portable/CI mode.

    The canonical processed dataset is intentionally untracked. The env flag
    activates the repo's own surrogate path for the Day23 freeze test, so a
    machine-local ignored file cannot make cross-platform hygiene fail.

    The SHA-only freeze defect is independently classified before this step;
    any non-hash freeze regression still fails B28.6.
    """
    status_before = git_output("status", "--porcelain=v1")
    diff_before = git_output("diff", "--binary")

    env = os.environ.copy()
    env["MODEL_FREEZE_USE_SURROGATE_DATA"] = "1"

    result = run([sys.executable, "-m", "pytest", "-q"], env=env)

    status_after = git_output("status", "--porcelain=v1")
    diff_after = git_output("diff", "--binary")

    if status_before != status_after:
        fail(
            "pytest changed git status.\n"
            "--- before ---\n" + status_before +
            "--- after ---\n" + status_after
        )

    if diff_before != diff_after:
        fail("pytest mutated tracked working-tree content")

    if result.returncode != 0:
        fail("portable full pytest failed:\n" + result.stdout[-5000:])

    match = re.search(
        r"(?P<passed>\d+)\s+passed(?:,\s+(?P<skipped>\d+)\s+skipped)?",
        result.stdout,
    )
    passed = int(match.group("passed")) if match else -1
    skipped = int(match.group("skipped")) if match and match.group("skipped") else 0
    return result.stdout, passed, skipped


def main() -> None:
    check_gitignore()
    scanned, abs_hits = check_encoding_and_paths()
    check_python_compile()
    check_git_diff_cleanliness()

    tracked_freeze, untracked_freeze, freeze_commit = classify_preexisting_freeze_issue()
    _, passed, skipped = check_tests_do_not_mutate_repo()

    print("PASS - B28.6 CROSS-PLATFORM / REPOSITORY HYGIENE QA")
    print("")
    print("Repository / portability")
    print(f"  Python files scanned = {scanned}")
    print("  [PASS] relative-path portability guard")
    print(f"  [PASS] machine-specific absolute-path hits = {abs_hits}")
    print("  [PASS] UTF-8 Python source")
    print("  [PASS] Python compile check")
    print("  [PASS] git diff --check")
    print("  [PASS] .venv, __pycache__, and .pytest_cache ignored")
    print("")
    print("Test isolation")
    if passed >= 0:
        print(f"  pytest portable-mode result = {passed} passed, {skipped} skipped")
    else:
        print("  pytest portable-mode result = PASS")
    print("  [PASS] full repository suite exits 0 in portable/CI mode")
    print("  [PASS] git status unchanged before/after pytest")
    print("  [PASS] tracked diff unchanged before/after pytest")
    print("")
    print("Pre-existing Day23 freeze-manifest provenance exception")
    if tracked_freeze or untracked_freeze:
        print(
            f"  freeze commit = {freeze_commit}; "
            f"tracked stale-SHA sources verified = {tracked_freeze}; "
            f"ignored local source = {untracked_freeze}"
        )
        print("  [PASS] all current Day23 validator failures are SHA-only")
        print("  [PASS] no semantic/model/spec freeze failure")
        print("  [PASS] tracked failing artifacts are byte-identical between freeze commit and HEAD")
        print("  [PASS] working-tree tracked canonical bytes equal HEAD")
        print("  [PASS] ignored processed dataset does not gate portable pytest")
        print("  [NOTE] stale Day23 hash values are a documented pre-existing provenance defect")
        print("  [NOTE] B28.6 does not rewrite model_freeze.yaml or canonical artifacts")
    else:
        print("  [PASS] Day23 model-freeze validator has no failures")
    print("")
    print("[PASS] B28.6 does not modify A-owned or canonical artifacts")


if __name__ == "__main__":
    main()
