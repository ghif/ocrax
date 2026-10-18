"""Stage and run the RxHandBD fine-tune on Colab through Google Colab CLI."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ocrax.infrastructure.data.manifests import validate_colab_manifests
from ocrax.infrastructure.cloud.colab import refresh_access_token, run_colab, split_file
from ocrax.infrastructure.cloud.gcs import parse_gcs, require_gcs_prefix


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "src" / "ocrax"
MANIFESTS = ROOT / "data" / "manifests"
REMOTE_DIR = "/content/handnotes-ocr"
REMOTE_ARCHIVE = "/content/handnotes-ocr.tar.gz"
REMOTE_CONFIG = "/content/colab-finetune-config.json"
REMOTE_STATUS = "/content/colab-finetune-status.json"
REMOTE_INPUT_PART = "/content/handnotes-ocr.tar.gz.part"
REMOTE_EXEC_TIMEOUT = "21600"
FINETUNED_RUNS_DIR = "finetuned-runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-gcs", default="gs://handnotes-ocr/model-dir")
    parser.add_argument("--checkpoint-gcs", help="Stored Orbax checkpoint prefix; defaults to MODEL_GCS/ppocrv6_medium_rec_orbax")
    parser.add_argument("--dataset-gcs", default="gs://all-datasets/RxHandBD")
    parser.add_argument(
        "--output-gcs",
        help="Deprecated destination override; must be MODEL_GCS/finetuned-runs. Defaults there.",
    )
    parser.add_argument("--gpu", choices=("T4", "L4", "G4", "A100", "H100"), default="T4")
    parser.add_argument("--epochs", type=int, default=3)
    return parser.parse_args()














def main() -> int:
    args = parse_args()
    colab = shutil.which("colab")
    if not colab:
        raise SystemExit("Colab CLI is not installed or `colab` is not on PATH.")
    for uri, option in (
        (args.model_gcs, "--model-gcs"),
        (args.dataset_gcs, "--dataset-gcs"),
    ):
        try:
            require_gcs_prefix(uri, option)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    model_bucket, model_prefix = require_gcs_prefix(args.model_gcs, "--model-gcs")
    output_base = f"gs://{model_bucket}/{model_prefix}/{FINETUNED_RUNS_DIR}"
    if args.output_gcs:
        try:
            requested_output = "gs://" + "/".join(
                part for part in parse_gcs(args.output_gcs) if part
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if requested_output.rstrip("/") != output_base:
            raise SystemExit(
                f"Colab checkpoints are always stored under {output_base}; "
                "omit --output-gcs or set it to that prefix."
            )
    checkpoint_gcs = args.checkpoint_gcs or args.model_gcs.rstrip("/") + "/ppocrv6_medium_rec_orbax"
    try:
        require_gcs_prefix(checkpoint_gcs, "--checkpoint-gcs")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.epochs < 1:
        raise SystemExit("--epochs must be a positive integer")
    try:
        validate_colab_manifests(MANIFESTS)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Invalid Colab training manifests: {exc}") from exc

    try:
        import google.auth
        from google.auth.transport.requests import Request
    except ImportError as exc:
        raise SystemExit("Install Google auth support in this environment: python -m pip install google-auth") from exc
    try:
        credentials, project = google.auth.default(scopes=[
            "openid",
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/colaboratory",
        ])
    except google.auth.exceptions.DefaultCredentialsError as exc:
        raise SystemExit(
            "Local Google ADC was not found. Run `gcloud auth application-default login "
            "--scopes=openid,https://www.googleapis.com/auth/cloud-platform,"
            "https://www.googleapis.com/auth/userinfo.email,"
            "https://www.googleapis.com/auth/colaboratory` on this Mac, then retry."
        ) from exc
    request = Request()
    try:
        # Validate ADC before allocating a Colab runtime. This also makes an
        # expired or incorrectly-scoped local login fail on the Mac, where the
        # user can complete the normal gcloud browser flow instead of being
        # presented with Colab's kernel-side verification-code prompt.
        credentials.refresh(request)
    except Exception as exc:
        raise SystemExit(
            "Could not refresh local Google ADC credentials. On this Mac, run:\n"
            "  gcloud auth application-default login "
            "--scopes=openid,https://www.googleapis.com/auth/cloud-platform,"
            "https://www.googleapis.com/auth/userinfo.email,"
            "https://www.googleapis.com/auth/colaboratory\n"
            f"Then retry. Details: {exc}"
        ) from exc
    run_id = datetime.now(timezone.utc).strftime("ppocrv6-rxhandbd-%Y%m%dT%H%M%S%fZ")
    session = run_id.lower()
    output_gcs = output_base + "/" + run_id
    config = {
        "model_gcs": args.model_gcs,
        "checkpoint_gcs": checkpoint_gcs,
        "dataset_gcs": args.dataset_gcs,
        "output_gcs": output_gcs,
        "gpu": args.gpu,
        "source_archive": REMOTE_ARCHIVE,
        "source_part_prefix": REMOTE_INPUT_PART,
        "source_dir": REMOTE_DIR,
        "run_id": run_id,
        "epochs": args.epochs,
        "phase": "prepare",
        "gcp_project": project,
        "quota_project_id": getattr(credentials, "quota_project_id", None),
    }

    with tempfile.TemporaryDirectory(prefix="ppocrv6-colab-") as temp_dir:
        temp = Path(temp_dir)
        archive_path = temp / "handnotes-ocr.tar.gz"
        session_config = temp / "sessions.json"
        remote_config_path = temp / "colab-finetune-config.json"
        with tarfile.open(archive_path, "w:gz") as archive:
            for source in sorted(PACKAGE.rglob("*.py")):
                archive.add(source, f"handnotes-ocr/src/ocrax/{source.relative_to(PACKAGE)}")
            for name in ("train.jsonl", "validation.jsonl", "split_metadata.json"):
                source = MANIFESTS / name
                archive.add(source, f"handnotes-ocr/data/manifests/{name}")
        input_parts = split_file(archive_path, temp / "input-parts", "handnotes-ocr.tar.gz.part")
        config["source_archive_parts"] = len(input_parts)

        session_may_exist = False
        try:
            # A client-side error can happen after Colab accepted the session
            # creation request. Attempt cleanup even when `new` exits nonzero.
            session_may_exist = True
            run_colab(colab, session_config, "new", "-s", session, "--gpu", args.gpu)
            remote_config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            for index, part in enumerate(input_parts):
                run_colab(
                    colab, session_config, "upload", "-s", session, str(part),
                    f"{REMOTE_INPUT_PART}{index:05d}",
                )
            run_colab(colab, session_config, "upload", "-s", session, str(remote_config_path), REMOTE_CONFIG)
            run_colab(
                colab, session_config, "exec", "-s", session,
                "-f", str(ROOT / "scripts" / "colab_remote.py"), "--timeout", REMOTE_EXEC_TIMEOUT,
            )
            local_status = temp / "colab-finetune-status.json"
            run_colab(colab, session_config, "download", "-s", session, REMOTE_STATUS, str(local_status))
            status = json.loads(local_status.read_text(encoding="utf-8"))
            if status.get("status") != "prepared":
                raise RuntimeError(f"Colab runtime preparation failed: {status.get('error', status)}")

            try:
                access_token, token_expiry = refresh_access_token(credentials, request)
            except Exception as exc:
                raise RuntimeError(f"Could not refresh local Google ADC credentials before GCS staging: {exc}") from exc
            config["phase"] = "stage"
            config["gcs_access_token"] = access_token
            config["gcs_token_expires_at"] = token_expiry
            remote_config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            run_colab(colab, session_config, "upload", "-s", session, str(remote_config_path), REMOTE_CONFIG)
            run_colab(
                colab, session_config, "exec", "-s", session,
                "-f", str(ROOT / "scripts" / "colab_remote.py"), "--timeout", REMOTE_EXEC_TIMEOUT,
            )
            run_colab(colab, session_config, "download", "-s", session, REMOTE_STATUS, str(local_status))
            status = json.loads(local_status.read_text(encoding="utf-8"))
            if status.get("status") != "staged":
                raise RuntimeError(f"GCS input staging did not complete: {status.get('error', status)}")

            # Training reads the staged local files only. Remove the GCS token
            # from the next uploaded config and release it before model work.
            config.pop("gcs_access_token", None)
            config.pop("gcs_token_expires_at", None)
            config["phase"] = "train"
            remote_config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            run_colab(colab, session_config, "upload", "-s", session, str(remote_config_path), REMOTE_CONFIG)
            run_colab(
                colab, session_config, "exec", "-s", session,
                "-f", str(ROOT / "scripts" / "colab_remote.py"), "--timeout", REMOTE_EXEC_TIMEOUT,
            )
            run_colab(colab, session_config, "download", "-s", session, REMOTE_STATUS, str(local_status))
            status = json.loads(local_status.read_text(encoding="utf-8"))
            if status.get("status") != "trained":
                raise RuntimeError(f"Remote Colab training did not complete: {status.get('error', status)}")
            if status.get("test_set_used") is not False:
                raise RuntimeError("Remote training metadata indicates the test set was used")

            # Training can outlast an OAuth access token. Mint a fresh token
            # only for the short final upload phase, after training is done.
            try:
                access_token, token_expiry = refresh_access_token(credentials, request)
            except Exception as exc:
                raise RuntimeError(f"Could not refresh local Google ADC credentials before output upload: {exc}") from exc
            config["phase"] = "upload"
            config["gcs_access_token"] = access_token
            config["gcs_token_expires_at"] = token_expiry
            remote_config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            run_colab(colab, session_config, "upload", "-s", session, str(remote_config_path), REMOTE_CONFIG)
            run_colab(
                colab, session_config, "exec", "-s", session,
                "-f", str(ROOT / "scripts" / "colab_remote.py"), "--timeout", "3300",
            )
            run_colab(colab, session_config, "download", "-s", session, REMOTE_STATUS, str(local_status))
            status = json.loads(local_status.read_text(encoding="utf-8"))
            if status.get("status") != "complete":
                raise RuntimeError(f"GCS output upload did not complete: {status.get('error', status)}")
            metadata = status
            print(json.dumps({
                "status": "complete",
                "output_gcs": output_gcs,
                "uploaded_files": metadata.get("checkpoint_files"),
                "best_epoch": metadata.get("best_epoch"),
                "best_validation": metadata.get("best_validation"),
                "baseline_validation": metadata.get("baseline_validation"),
                "history": metadata.get("history"),
                "backend": metadata.get("backend"),
                "gcs_inputs": {"model": args.model_gcs, "checkpoint": checkpoint_gcs, "dataset": args.dataset_gcs},
            }, indent=2), flush=True)
        finally:
            if session_may_exist:
                try:
                    run_colab(colab, session_config, "stop", "-s", session)
                except Exception as exc:
                    print(f"Warning: failed to stop Colab session {session}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
