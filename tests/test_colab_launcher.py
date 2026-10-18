from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import google.auth

from ocrax.cli import colab as colab_launcher


class FakeCredentials:
    quota_project_id = "quota-project"
    token = None
    expiry = None
    refresh_count = 0

    def refresh(self, _request) -> None:
        self.refresh_count += 1
        self.token = f"fake-access-token-{self.refresh_count}"
        self.expiry = datetime.now(timezone.utc) + timedelta(hours=1)


class ColabLauncherTests(unittest.TestCase):
    def test_main_runs_phases_and_stops_session(self) -> None:
        phases: list[str] = []
        commands: list[tuple[str, ...]] = []
        uploaded_configs: dict[str, dict] = {}
        uploaded_archive_parts: list[tuple[int, bytes]] = []
        statuses: dict[str, dict] = {
            "prepare": {"status": "prepared", "backend": "gpu"},
            "stage": {"status": "staged", "staged_inputs": {"images": 2}},
            "train": {"status": "trained", "test_set_used": False},
            "upload": {"status": "complete", "checkpoint_files": 3},
        }
        active_phase = "prepare"
        stopped = False

        def fake_run_colab(_colab: str, _session_config: Path, *args: str) -> None:
            nonlocal active_phase, stopped
            commands.append(tuple(args))
            operation = args[0]
            local_upload = Path(args[-2]) if operation == "upload" else None
            if local_upload and local_upload.name.startswith("handnotes-ocr.tar.gz.part"):
                part_index = int(local_upload.name.removeprefix("handnotes-ocr.tar.gz.part"))
                uploaded_archive_parts.append((part_index, local_upload.read_bytes()))
            if operation == "upload" and Path(args[-2]).name == "colab-finetune-config.json":
                uploaded_config = json.loads(Path(args[-2]).read_text(encoding="utf-8"))
                active_phase = uploaded_config["phase"]
                uploaded_configs[active_phase] = uploaded_config
            elif operation == "exec":
                phases.append(active_phase)
            elif operation == "download":
                Path(args[-1]).write_text(json.dumps(statuses[active_phase]), encoding="utf-8")
            elif operation == "stop":
                stopped = True

        credentials = FakeCredentials()
        with (
            patch.object(colab_launcher.shutil, "which", return_value="/usr/bin/colab"),
            patch.object(colab_launcher, "run_colab", side_effect=fake_run_colab),
            patch.object(google.auth, "default", return_value=(credentials, "quota-project")),
            patch.object(sys, "argv", [
                "colab_launcher",
                "--model-gcs", "gs://models/base",
                "--dataset-gcs", "gs://datasets/RxHandBD",
                "--output-gcs", "gs://models/base/finetuned-runs",
                "--epochs", "2",
            ]),
        ):
            self.assertEqual(colab_launcher.main(), 0)

        self.assertEqual(phases, ["prepare", "stage", "train", "upload"])
        self.assertEqual(credentials.refresh_count, 3)
        self.assertIn("gcs_access_token", uploaded_configs["stage"])
        self.assertNotIn("gcs_access_token", uploaded_configs["train"])
        self.assertIn("gcs_access_token", uploaded_configs["upload"])
        self.assertNotEqual(
            uploaded_configs["stage"]["gcs_access_token"],
            uploaded_configs["upload"]["gcs_access_token"],
        )
        self.assertTrue(stopped)
        self.assertEqual(
            [command[0] for command in commands if command[0] in {"new", "exec", "stop"}],
            ["new", "exec", "exec", "exec", "exec", "stop"],
        )
        self.assertFalse(any(command[0] == "auth" for command in commands))
        archive_bytes = b"".join(part for _, part in sorted(uploaded_archive_parts))
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            archived_names = set(archive.getnames())
        expected_source_names = {
            f"handnotes-ocr/src/ocrax/{source.relative_to(colab_launcher.PACKAGE)}"
            for source in colab_launcher.PACKAGE.rglob("*.py")
        }
        expected_manifest_names = {
            "handnotes-ocr/data/manifests/train.jsonl",
            "handnotes-ocr/data/manifests/validation.jsonl",
            "handnotes-ocr/data/manifests/split_metadata.json",
        }
        self.assertEqual(archived_names, expected_source_names | expected_manifest_names)

    def test_main_attempts_cleanup_if_session_creation_errors(self) -> None:
        operations: list[str] = []

        def fake_run_colab(_colab: str, _session_config: Path, *args: str) -> None:
            operations.append(args[0])
            if args[0] == "new":
                raise subprocess.CalledProcessError(1, ["colab", *args])

        with (
            patch.object(colab_launcher.shutil, "which", return_value="/usr/bin/colab"),
            patch.object(colab_launcher, "run_colab", side_effect=fake_run_colab),
            patch.object(google.auth, "default", return_value=(FakeCredentials(), "quota-project")),
            patch.object(sys, "argv", [
                "colab_launcher",
                "--model-gcs", "gs://models/base",
                "--dataset-gcs", "gs://datasets/RxHandBD",
                "--output-gcs", "gs://models/base/finetuned-runs",
            ]),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                colab_launcher.main()

        self.assertEqual(operations, ["new", "stop"])

    def test_cleanup_error_does_not_hide_session_creation_error(self) -> None:
        operations: list[str] = []

        def fake_run_colab(_colab: str, _session_config: Path, *args: str) -> None:
            operations.append(args[0])
            if args[0] == "new":
                raise subprocess.CalledProcessError(1, ["colab", *args])
            if args[0] == "stop":
                raise OSError("colab executable disappeared")

        with (
            patch.object(colab_launcher.shutil, "which", return_value="/usr/bin/colab"),
            patch.object(colab_launcher, "run_colab", side_effect=fake_run_colab),
            patch.object(google.auth, "default", return_value=(FakeCredentials(), "quota-project")),
            patch.object(sys, "argv", [
                "colab_launcher",
                "--model-gcs", "gs://models/base",
                "--dataset-gcs", "gs://datasets/RxHandBD",
                "--output-gcs", "gs://models/base/finetuned-runs",
            ]),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                colab_launcher.main()

        self.assertEqual(operations, ["new", "stop"])


if __name__ == "__main__":
    unittest.main()
