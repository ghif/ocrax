"""Remote Colab bootstrap, executed by `colab exec -f`."""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path


CONFIG_PATH = Path("/content/colab-finetune-config.json")
STATUS_PATH = Path("/content/colab-finetune-status.json")
CONTENT_DIR = Path("/content")
INPUTS_DIR = CONTENT_DIR / "inputs"
OUTPUT_DIR = Path("/content/fine-tuned/ppocrv6_medium_rec_rxhandbd_finetuned")
PINNED_PACKAGES = [
    "jax[cuda12]==0.9.0",
    "flax==0.12.3",
    "optax==0.2.3",
    "orbax-checkpoint==0.11.32",
    "numpy==2.3.5",
    "Pillow==12.1.0",
    "PyYAML==6.0.2",
    "google-cloud-storage>=2.18,<4",
]


def run(command: list[str], *, cwd: Path | None = None, env=None) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=cwd, env=env)


def capture(command: list[str]) -> str:
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    return result.stdout


def join_parts(prefix: str, count: int, destination: Path) -> None:
    with destination.open("wb") as output:
        for index in range(count):
            part = Path(f"{prefix}{index:05d}")
            if not part.is_file():
                raise FileNotFoundError(f"Missing uploaded archive part: {part}")
            with part.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)


def parse_gcs(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected a GCS URI, got {uri!r}")
    bucket, _, prefix = uri[5:].partition("/")
    if not bucket:
        raise ValueError(f"Invalid GCS URI: {uri!r}")
    return bucket, prefix.strip("/")


def parse_token_expiry(expiry_text: str) -> datetime:
    expiry = datetime.fromisoformat(expiry_text)
    # google-auth can expose an expiry without tzinfo for some ADC credential
    # types. Google token expiry timestamps are UTC, so normalize before math.
    if expiry.tzinfo is None:
        return expiry.replace(tzinfo=timezone.utc)
    return expiry.astimezone(timezone.utc)


def token_seconds_remaining(expiry_text: str) -> float:
    expiry = parse_token_expiry(expiry_text)
    return (expiry - datetime.now(timezone.utc)).total_seconds()


def download_prefix(client, uri: str, destination: Path, *, exclude_prefixes: tuple[str, ...] = ()) -> int:
    bucket_name, prefix = parse_gcs(uri)
    query = prefix.rstrip("/") + "/" if prefix else ""
    excluded = tuple(item.strip("/") + "/" for item in exclude_prefixes)
    blobs = [
        blob for blob in client.list_blobs(bucket_name, prefix=query)
        if blob.name
        and not blob.name.endswith("/")
        and not any(blob.name[len(query):].startswith(item) for item in excluded)
    ]
    if not blobs:
        raise FileNotFoundError(f"No objects found under {uri}")
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    for blob in blobs:
        target = (destination / blob.name[len(query):]).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Unsafe object path under {uri}: {blob.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(target))
    return len(blobs)


def stage_gcs_inputs(client, config: dict, repo_dir: Path) -> dict:
    inputs = INPUTS_DIR
    model_dir = inputs / "model-dir"
    print(f"Reading model from {config['model_gcs']} directly on Colab...", flush=True)
    model_count = download_prefix(
        client,
        config["model_gcs"],
        model_dir,
        exclude_prefixes=("finetuned-runs",),
    )
    if not (model_dir / "inference.yml").is_file():
        raise FileNotFoundError(f"Expected inference.yml under {config['model_gcs']}")
    checkpoint_uri = config["checkpoint_gcs"]
    default_uri = config["model_gcs"].rstrip("/") + "/ppocrv6_medium_rec_orbax"
    checkpoint_dir = model_dir / "ppocrv6_medium_rec_orbax" if checkpoint_uri.rstrip("/") == default_uri.rstrip("/") else inputs / "checkpoint"
    checkpoint_count = 0
    if checkpoint_dir != model_dir / "ppocrv6_medium_rec_orbax":
        print(f"Reading checkpoint from {checkpoint_uri} directly on Colab...", flush=True)
        checkpoint_count = download_prefix(client, checkpoint_uri, checkpoint_dir)
    if not (checkpoint_dir / "manifest.ocdbt").is_file():
        raise FileNotFoundError(f"Expected Orbax manifest under {checkpoint_uri}")

    dataset_bucket, dataset_prefix = parse_gcs(config["dataset_gcs"])
    image_root = inputs / "RxHandBD" / "Train_Set"
    staged_images: dict[str, str] = {}
    for split in ("train", "validation"):
        source = repo_dir / "data" / "manifests" / f"{split}.jsonl"
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            raise ValueError(f"Empty manifest: {source}")
        for record in records:
            filename = Path(record["image"]).name
            if not filename or filename in (".", ".."):
                raise ValueError(f"Invalid image path: {record['image']!r}")
            previous_source = staged_images.get(filename)
            if previous_source is not None and previous_source != record["image"]:
                raise ValueError(
                    f"Image basename collision while staging GCS data: {previous_source!r} "
                    f"and {record['image']!r} both map to {filename!r}"
                )
            target = image_root / filename
            if previous_source is None:
                object_name = "/".join(part for part in (dataset_prefix, "Train_Set", filename) if part)
                target.parent.mkdir(parents=True, exist_ok=True)
                client.bucket(dataset_bucket).blob(object_name).download_to_filename(str(target))
                staged_images[filename] = record["image"]
            record["image"] = str(target)
        (repo_dir / "data" / "manifests" / f"colab_{split}.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    return {"model_dir": str(model_dir), "checkpoint_dir": str(checkpoint_dir), "model_files": model_count, "checkpoint_files": checkpoint_count, "images": len(staged_images)}


def gcs_client_from_config(config: dict):
    access_token = config.pop("gcs_access_token", None)
    if not access_token:
        raise RuntimeError("Missing short-lived GCS access token in remote config")
    expiry_text = config.pop("gcs_token_expires_at", None)
    if not expiry_text:
        raise RuntimeError("Missing GCS access token expiry in remote config")
    expiry = parse_token_expiry(expiry_text)
    if (expiry - datetime.now(timezone.utc)).total_seconds() < 900:
        raise RuntimeError("GCS access token expires in under 15 minutes; refresh local ADC and retry")
    from google.cloud import storage
    from google.oauth2.credentials import Credentials
    client = storage.Client(
        project=config.get("gcp_project"),
        credentials=Credentials(
            token=access_token,
            # google-auth currently compares this value to a naive UTC
            # datetime in Credentials.expired. Keep timezone-aware values for
            # our own calculations above, then pass naive UTC for compatibility.
            expiry=expiry.replace(tzinfo=None),
            quota_project_id=config.get("quota_project_id"),
        ),
    )
    del access_token
    return client


def upload_checkpoint(gcs, config: dict) -> dict:
    output_dir = OUTPUT_DIR
    metadata_path = output_dir / "finetuning.json"
    if not (output_dir / "manifest.ocdbt").is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Trained checkpoint is missing its Orbax manifest or finetuning metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("test_set_used") is not False:
        raise RuntimeError("Training metadata indicates the test set was used")
    output_bucket, output_prefix = parse_gcs(config["output_gcs"])
    uploaded = 0
    uploaded_names: set[str] = set()
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            relative_name = path.relative_to(output_dir).as_posix()
            object_name = "/".join(part for part in (output_prefix.rstrip("/"), relative_name) if part)
            # The create-only precondition prevents overwrites without
            # requiring storage.objects.list on the output bucket.
            gcs.bucket(output_bucket).blob(object_name).upload_from_filename(
                str(path), if_generation_match=0,
            )
            uploaded += 1
            uploaded_names.add(relative_name)
    if "manifest.ocdbt" not in uploaded_names or "finetuning.json" not in uploaded_names:
        raise RuntimeError("Checkpoint upload did not include the Orbax manifest and finetuning metadata")
    return {
        "status": "complete",
        "checkpoint_files": uploaded,
        "best_epoch": metadata.get("best_epoch"),
        "best_validation": metadata.get("best_validation"),
        "baseline_validation": metadata.get("baseline_validation"),
        "history": metadata.get("history"),
        "backend": metadata.get("backend"),
        "test_set_used": metadata.get("test_set_used"),
    }


def prepare_runtime(config: dict) -> dict:
    # Prepare the runtime and source before the launcher mints a GCS token.
    run([sys.executable, "-m", "pip", "install", "--upgrade", *PINNED_PACKAGES])
    for distribution in ("jax-cuda13-plugin", "jax-cuda13-pjrt"):
        installed = subprocess.run(
            [sys.executable, "-m", "pip", "show", distribution],
            text=True, capture_output=True,
        )
        if installed.returncode == 0:
            run([sys.executable, "-m", "pip", "uninstall", "--yes", distribution])

    probe = capture([sys.executable, "-c", (
        "import json, jax, numpy as np, sys; devices=jax.devices(); "
        "print(json.dumps({'python': sys.version, 'backend': jax.default_backend(), "
        "'devices': [str(d) for d in devices], "
        "'device_kinds': [getattr(d, 'device_kind', 'unknown') for d in devices], "
        "'numpy': np.__version__}))"
    )])
    probe_data = json.loads(probe.strip().splitlines()[-1])
    if probe_data["backend"] != "gpu":
        raise RuntimeError(f"Expected a CUDA GPU JAX backend, got {probe_data['backend']!r}")

    source_archive = Path(config["source_archive"])
    join_parts(config["source_part_prefix"], int(config["source_archive_parts"]), source_archive)
    source_root = CONTENT_DIR.resolve()
    with tarfile.open(source_archive, "r:gz") as archive:
        for member in archive.getmembers():
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported source archive member type: {member.name}")
            target = (source_root / member.name).resolve()
            if not target.is_relative_to(source_root):
                raise ValueError(f"Unsafe source archive member: {member.name}")
        archive.extractall(source_root, filter="data")
    return {"status": "prepared", "backend": probe_data["backend"], "devices": probe_data["devices"]}


def validate_staged_inputs(config: dict, repo_dir: Path) -> tuple[Path, Path, Path, Path]:
    model_dir = INPUTS_DIR / "model-dir"
    if not (model_dir / "inference.yml").is_file():
        raise FileNotFoundError(f"Expected model config at {model_dir / 'inference.yml'}")
    checkpoint_uri = config["checkpoint_gcs"]
    default_uri = config["model_gcs"].rstrip("/") + "/ppocrv6_medium_rec_orbax"
    checkpoint = (
        model_dir / "ppocrv6_medium_rec_orbax"
        if checkpoint_uri.rstrip("/") == default_uri.rstrip("/")
        else INPUTS_DIR / "checkpoint"
    )
    if not checkpoint.is_dir() or not (checkpoint / "manifest.ocdbt").is_file():
        raise FileNotFoundError(f"Expected staged Orbax checkpoint at {checkpoint}")
    train_manifest = repo_dir / "data" / "manifests" / "colab_train.jsonl"
    validation_manifest = repo_dir / "data" / "manifests" / "colab_validation.jsonl"
    for manifest in (train_manifest, validation_manifest):
        if not manifest.is_file():
            raise FileNotFoundError(f"Staged manifest missing: {manifest}")
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                image_path = Path(json.loads(line)["image"])
                if not image_path.is_file():
                    raise FileNotFoundError(f"Manifest image missing on Colab: {image_path}")
    return model_dir, checkpoint, train_manifest, validation_manifest


def stage_inputs(config: dict) -> dict:
    repo_dir = Path(config["source_dir"])
    if not (repo_dir / "src" / "ocrax" / "cli" / "train.py").is_file():
        raise FileNotFoundError("Prepared training source is missing; rerun runtime preparation")
    gcs = gcs_client_from_config(config)
    staged = stage_gcs_inputs(gcs, config, repo_dir)
    validate_staged_inputs(config, repo_dir)
    return {"status": "staged", "staged_inputs": staged}


def train_staged(config: dict) -> dict:
    repo_dir = Path(config["source_dir"])
    if not (repo_dir / "src" / "ocrax" / "cli" / "train.py").is_file():
        raise FileNotFoundError("Prepared training source is missing; rerun runtime preparation")
    model_dir, checkpoint, train_manifest, validation_manifest = validate_staged_inputs(config, repo_dir)
    print(json.dumps({"model_dir": str(model_dir), "checkpoint": str(checkpoint), "train_manifest": str(train_manifest), "validation_manifest": str(validation_manifest)}), flush=True)
    output_dir = OUTPUT_DIR
    command = [
        sys.executable, "-m", "ocrax.cli.train",
        "--model-dir", str(model_dir),
        "--checkpoint", str(checkpoint),
        "--train-manifest", str(train_manifest),
        "--validation-manifest", str(validation_manifest),
        "--output", str(output_dir),
        "--epochs", str(config["epochs"]),
        "--batch-size", "8",
        "--learning-rate", "1e-5",
        "--seed", "20261017",
    ]
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_dir / "src") + os.pathsep + env.get("PYTHONPATH", "")
    run(command, cwd=repo_dir, env=env)

    metadata_path = output_dir / "finetuning.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Trainer did not produce {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("test_set_used") is not False:
        raise RuntimeError("Trainer metadata indicates the test set was used")
    if (metadata.get("epochs_requested"), metadata.get("batch_size"), metadata.get("seed")) != (config["epochs"], 8, 20261017):
        raise RuntimeError("Trainer metadata does not match the requested training recipe")

    summary = {
        "status": "trained",
        "checkpoint_files": sum(1 for path in output_dir.rglob("*") if path.is_file()),
        "best_epoch": metadata.get("best_epoch"),
        "best_validation": metadata.get("best_validation"),
        "baseline_validation": metadata.get("baseline_validation"),
        "history": metadata.get("history"),
        "backend": metadata.get("backend"),
        "test_set_used": metadata.get("test_set_used"),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> dict:
    # The launcher passes a short-lived OAuth access token derived from local
    # ADC. Remove the uploaded config immediately; never print it.
    raw_config = CONFIG_PATH.read_text(encoding="utf-8")
    CONFIG_PATH.unlink(missing_ok=True)
    config = json.loads(raw_config)
    del raw_config
    phase = config.get("phase", "train")
    if phase == "prepare":
        return prepare_runtime(config)
    if phase == "stage":
        return stage_inputs(config)
    if phase == "upload":
        return upload_checkpoint(gcs_client_from_config(config), config)
    if phase != "train":
        raise ValueError(f"Unknown Colab run phase: {phase!r}")
    return train_staged(config)


if __name__ == "__main__":
    try:
        STATUS_PATH.unlink(missing_ok=True)
        result = main()
        STATUS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except BaseException as exc:
        STATUS_PATH.write_text(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n", encoding="utf-8")
        raise
