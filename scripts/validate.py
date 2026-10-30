"""Validate the installed project, dependencies, and test suites."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="Also require a Git repository with a clean working tree.",
    )
    args = parser.parse_args(argv)

    environment = os.environ.copy()
    for name in tuple(environment):
        if name in {"PYTHONPATH", "PYTHONHOME", "FASHION_MNIST_PROJECT_ROOT"}:
            environment.pop(name)
        elif name.startswith((
            "MLFLOW_", "MINIO_", "MODEL_", "WEBHOOK_",
            "TRAINING_WEBHOOK_", "FASTAPI_",
            "GIT_", "PYTEST_",
        )):
            environment.pop(name)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence = ROOT / "artifacts/validation" / f"{stamp}-{uuid.uuid4().hex[:8]}"
    evidence.mkdir(parents=True)
    results = []

    def run(name: str, command: list[str]) -> bool:
        print(f"\n{name}", flush=True)
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                check=False,
            )
            passed = completed.returncode == 0
            result = {
                "name": name,
                "passed": passed,
                "returncode": completed.returncode,
            }
        except OSError as error:
            passed = False
            result = {
                "name": name,
                "passed": False,
                "error": str(error),
            }
            print(error)
        results.append(result)
        print(f"{'PASS' if passed else 'FAIL'}: {name}", flush=True)
        return passed

    probe = """
from pathlib import Path
import sys
from fashion_mnist_mlops import train, api, inference, storage, tracking, ui
from fashion_mnist_mlops.paths import get_project_root

root = Path(sys.argv[1]).resolve()
assert get_project_root() == root

for module in (train, api, inference, storage, tracking, ui):
    expected = root / "src/fashion_mnist_mlops"
    assert Path(module.__file__).resolve().is_relative_to(expected), module.__file__

print("Installed modules resolve to this project.")
"""

    installed = run(
        "Installed package location",
        [sys.executable, "-I", "-c", probe, str(ROOT)],
    )

    run(
        "Python syntax",
        [sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"],
    )
    run(
        "Dependency consistency",
        [sys.executable, "-m", "pip", "check"],
    )

    if installed:
        run(
            "Unit tests",
            [
                sys.executable, "-m", "pytest", "-q", "-rs",
                "tests/unit",
                f"--junitxml={evidence / 'unit.xml'}",
            ],
        )
        integration_xml = evidence / "integration.xml"
        integration_passed = run(
            "Integration tests",
            [
                sys.executable, "-m", "pytest", "-q", "-rs",
                "tests/integration",
                f"--junitxml={integration_xml}",
            ],
        )

        complete = False
        details = {}
        if integration_xml.is_file():
            try:
                tree = ET.parse(integration_xml)
                cases = list(tree.iter("testcase"))
                skipped = sum(case.find("skipped") is not None for case in cases)
                failed = sum(
                    case.find("failure") is not None
                    or case.find("error") is not None
                    for case in cases
                )
                complete = (
                    integration_passed
                    and len(cases) > 0
                    and skipped == 0
                    and failed == 0
                )
                details = {
                    "test_cases": len(cases),
                    "skipped": skipped,
                    "failed": failed,
                }
            except (ET.ParseError, OSError) as error:
                details = {"error": str(error)}

        results.append({
            "name": "Integration completion without skips",
            "passed": complete,
            **details,
        })
        print(
            f"{'PASS' if complete else 'FAIL'}: "
            "Integration completion without skips",
            flush=True,
        )
    else:
        results.append({
            "name": "Test execution",
            "passed": False,
            "error": "Tests were not run because the installed package is incorrect.",
        })

    if args.require_clean:
        try:
            repository = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            reported_root = repository.stdout.strip()
            git_ok = (
                repository.returncode == 0
                and bool(reported_root)
                and Path(reported_root).resolve() == ROOT.resolve()
            )
            git_details = {
                "returncode": repository.returncode,
                "reported_root": reported_root,
            }
        except (OSError, ValueError) as error:
            git_ok = False
            git_details = {"error": str(error)}
        results.append({
            "name": "Git repository location",
            "passed": git_ok,
            **git_details,
        })
        print(f"{'PASS' if git_ok else 'FAIL'}: Git repository location")
        if git_ok:
            run("Working-tree whitespace", ["git", "diff", "--check"])
            run("Staged whitespace", ["git", "diff", "--cached", "--check"])
            try:
                status = subprocess.run(
                    ["git", "status", "--porcelain", "--untracked-files=all"],
                    cwd=ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                clean = status.returncode == 0 and not status.stdout.strip()
                status_details = {"returncode": status.returncode}
                if status.stdout:
                    print(status.stdout)
                if status.stderr:
                    print(status.stderr)
            except OSError as error:
                clean = False
                status_details = {"error": str(error)}
            results.append({
                "name": "Clean Git working tree",
                "passed": clean,
                **status_details,
            })
            print(f"{'PASS' if clean else 'FAIL'}: Clean Git working tree")

    passed = all(item["passed"] for item in results)
    report = {
        "project": str(ROOT),
        "python": sys.executable,
        "passed": passed,
        "results": results,
    }
    report_path = evidence / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nValidation report: {report_path}")
    print("PROJECT VALIDATION PASSED" if passed else "PROJECT VALIDATION FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
