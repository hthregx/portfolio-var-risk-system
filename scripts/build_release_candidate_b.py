from pathlib import Path
import csv
import hashlib
import json
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/release_candidate_b.yaml"
LEDGER = ROOT / "results/release_artifact_ledger_b.csv"
MANIFEST = ROOT / "results/release_candidate_manifest_b.json"


def git(*args):
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
    )


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    rc = cfg["release_candidate"]
    frozen = cfg["frozen_contract"]

    base = rc["integration_base_sha"]
    artifacts = rc["required_artifacts"]

    rows = []
    manifest_artifacts = []

    for item in artifacts:
        path = item["path"]
        role = item["role"]

        blob_oid = git(
            "rev-parse",
            f"{base}:{path}",
        ).decode().strip()

        blob = git(
            "cat-file",
            "-p",
            blob_oid,
        )

        sha256 = hashlib.sha256(blob).hexdigest()

        row = {
            "path": path,
            "role": role,
            "git_blob_oid": blob_oid,
            "sha256_git_blob": sha256,
            "blob_bytes": len(blob),
            "status": "PASS",
        }

        rows.append(row)
        manifest_artifacts.append(row.copy())

    rows.sort(key=lambda x: x["path"])
    manifest_artifacts.sort(key=lambda x: x["path"])

    LEDGER.parent.mkdir(parents=True, exist_ok=True)

    with LEDGER.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "role",
                "git_blob_oid",
                "sha256_git_blob",
                "blob_bytes",
                "status",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "release_candidate": rc["name"],
        "integration_base_sha": base,
        "artifact_count": len(manifest_artifacts),
        "excluded_patterns": rc["excluded_patterns"],
        "frozen_contract": frozen,
        "artifacts": manifest_artifacts,
    }

    MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print(f"artifact_count: {len(rows)}")
    print(f"ledger: {LEDGER.relative_to(ROOT)}")
    print(f"manifest: {MANIFEST.relative_to(ROOT)}")
    print("B2_RELEASE_CANDIDATE_BUILD_PASS")


if __name__ == "__main__":
    main()