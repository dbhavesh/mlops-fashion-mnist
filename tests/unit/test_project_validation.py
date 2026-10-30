"""Failure-path coverage for the consolidated validation command."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[2] / "scripts/validate.py"
SPEC = importlib.util.spec_from_file_location("project_validation", SOURCE)
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


class TestProjectValidation(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.parent = Path(directory.name).resolve()
        self.root = self.parent / "project"
        self.root.mkdir()
        self.calls = []
        self.failures = {}
        self.errors = set()
        self.xml = '<testsuites><testsuite><testcase name="training"/></testsuite></testsuites>'
        self.repository = str(self.root)
        self.status = ""
        self.real_git = False
        self.original_run = subprocess.run
        replacement = patch.object(validation, "ROOT", self.root)
        replacement.start()
        self.addCleanup(replacement.stop)

    def invoke(self, command, **kwargs):
        self.calls.append((list(command), kwargs))
        if command[0] == "git":
            key = "git:" + " ".join(command[1:])
            output = self.repository if command[1] == "rev-parse" else self.status
            if command[1] == "diff":
                output = ""
            if self.real_git:
                return self.original_run(command, **kwargs)
        else:
            key = (
                "installed" if "-I" in command
                else "syntax" if "compileall" in command
                else "dependencies" if "pip" in command
                else "unit" if "tests/unit" in command
                else "integration"
            )
            output = ""
        if key in self.errors:
            raise OSError("Executable unavailable")
        if key == "integration" and self.xml is not None:
            filename = next(value.split("=", 1)[1] for value in command if value.startswith("--junitxml="))
            Path(filename).write_text(self.xml, encoding="utf-8")
        return subprocess.CompletedProcess(command, self.failures.get(key, 0), output, "")

    def execute(self, *arguments):
        with patch.object(validation.subprocess, "run", side_effect=self.invoke):
            with redirect_stdout(io.StringIO()):
                code = validation.main(list(arguments))
        reports = list((self.root / "artifacts/validation").glob("*/report.json"))
        report = json.loads(max(reports, key=lambda value: value.stat().st_mtime_ns).read_text())
        self.assertEqual(report["passed"], code == 0)
        self.assertEqual(report["project"], str(self.root))
        self.assertEqual(report["python"], sys.executable)
        return code, report

    def test_success_uses_active_python_and_both_suites(self):
        code, report = self.execute()
        self.assertEqual(code, 0)
        commands = [call[0] for call in self.calls]
        self.assertEqual(len(commands), 5)
        self.assertTrue(all(command[0] == sys.executable for command in commands))
        self.assertIn("tests/unit", commands[-2])
        self.assertIn("tests/integration", commands[-1])
        self.assertTrue(all(item["passed"] for item in report["results"]))

    def test_wrong_installation_prevents_test_execution(self):
        self.failures["installed"] = 1
        self.assertEqual(self.execute()[0], 1)
        self.assertFalse(any("pytest" in command for command, _ in self.calls))

    def test_syntax_failure_is_preserved_while_other_checks_continue(self):
        self.failures["syntax"] = 1
        self.assertEqual(self.execute()[0], 1)
        self.assertTrue(any("tests/integration" in command for command, _ in self.calls))

    def test_dependency_failure_is_preserved(self):
        self.failures["dependencies"] = 1
        self.assertEqual(self.execute()[0], 1)

    def test_unit_failure_is_preserved(self):
        self.failures["unit"] = 1
        self.assertEqual(self.execute()[0], 1)

    def test_integration_exit_failure_overrides_passing_xml(self):
        self.failures["integration"] = 1
        self.assertEqual(self.execute()[0], 1)

    def test_missing_integration_report_fails(self):
        self.xml = None
        self.assertEqual(self.execute()[0], 1)

    def test_empty_integration_report_fails(self):
        self.xml = "<testsuites/>"
        self.assertEqual(self.execute()[0], 1)

    def test_malformed_integration_report_fails(self):
        self.xml = "<testsuites"
        self.assertEqual(self.execute()[0], 1)

    def test_incomplete_integration_cases_fail(self):
        for marker in ("skipped", "failure", "error"):
            with self.subTest(marker=marker):
                self.xml = f'<testsuites><testsuite><testcase><{marker}/></testcase></testsuite></testsuites>'
                self.assertEqual(self.execute()[0], 1)

    def test_command_launch_error_is_reported(self):
        self.errors.add("syntax")
        code, report = self.execute()
        self.assertEqual(code, 1)
        self.assertTrue(any("error" in item for item in report["results"]))

    def test_exact_clean_repository_passes(self):
        self.assertEqual(self.execute("--require-clean")[0], 0)

    def test_parent_repository_is_rejected(self):
        self.repository = str(self.parent)
        self.assertEqual(self.execute("--require-clean")[0], 1)
        self.assertFalse(any(command[:2] == ["git", "status"] for command, _ in self.calls))

    def test_empty_repository_output_is_rejected(self):
        self.repository = ""
        self.assertEqual(self.execute("--require-clean")[0], 1)

    def test_git_failure_and_missing_executable_are_reported(self):
        key = "git:rev-parse --show-toplevel"
        self.failures[key] = 128
        self.assertEqual(self.execute("--require-clean")[0], 1)
        self.errors.add(key)
        self.assertEqual(self.execute("--require-clean")[0], 1)

    def test_staged_unstaged_and_untracked_changes_are_rejected(self):
        for status in ("A  README.md\n", " M README.md\n", "?? new.py\n"):
            with self.subTest(status=status):
                self.status = status
                self.assertEqual(self.execute("--require-clean")[0], 1)

    def test_status_error_cannot_look_clean(self):
        key = "git:status --porcelain --untracked-files=all"
        self.failures[key] = 128
        self.assertEqual(self.execute("--require-clean")[0], 1)
        self.errors.add(key)
        self.assertEqual(self.execute("--require-clean")[0], 1)

    def test_whitespace_failures_are_preserved(self):
        for key in ("git:diff --check", "git:diff --cached --check"):
            with self.subTest(command=key):
                self.failures = {key: 2}
                self.assertEqual(self.execute("--require-clean")[0], 1)

    def test_child_environment_isolated_without_changing_parent(self):
        overrides = {
            "GIT_DIR": str(self.parent / "other/.git"),
            "GIT_WORK_TREE": str(self.parent / "other"),
            "GIT_INDEX_FILE": str(self.parent / "other-index"),
            "PYTEST_ADDOPTS": "-k nonexistent",
            "PYTEST_PLUGINS": "unrelated_plugin",
            "PYTHONPATH": str(self.parent),
            "PYTHONHOME": str(self.parent),
            "FASHION_MNIST_PROJECT_ROOT": str(self.parent),
            "WEBHOOK_URL": "http://localhost:8765/webhook",
            "TRAINING_WEBHOOK_URL": "http://localhost:8765/webhook",
            "MINIO_ENDPOINT": "localhost:9000",
            "MLFLOW_TRACKING_URI": "http://localhost:5000",
            "MODEL_OBJECT_KEY": "model/example/model.pt",
            "FASTAPI_BASE_URL": "http://localhost:8000",
        }
        with patch.dict(os.environ, overrides):
            self.assertEqual(self.execute("--require-clean")[0], 0)
            for _, options in self.calls:
                self.assertFalse(set(overrides) & options["env"].keys())
            self.assertTrue(all(os.environ[key] == value for key, value in overrides.items()))

    @unittest.skipUnless(shutil.which("git"), "Git executable unavailable")
    def test_real_parent_repository_is_rejected(self):
        environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        self.original_run(["git", "init", "-q", str(self.parent)], env=environment, check=True)
        self.real_git = True
        self.assertEqual(self.execute("--require-clean")[0], 1)


if __name__ == "__main__":
    unittest.main()
