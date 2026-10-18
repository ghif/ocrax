"""Application workflow for CTC fine-tuning."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax import nnx
from PIL import Image, ImageEnhance

from ocrax.domain.metrics import edit_distance
from ocrax.infrastructure.checkpoints.orbax import restore_nnx, save_state
from ocrax.infrastructure.data.images import resize_rec_image
from ocrax.infrastructure.data.manifests import load_manifest, manifest_hash, validate_label_vocabulary
from ocrax.infrastructure.models.ppocrv6 import ctc_greedy_decode

def character_table(inference_yaml: Path) -> tuple[list[str], dict[str, int]]:
    config = yaml.safe_load(inference_yaml.read_text(encoding="utf-8"))
    chars = list(config["PostProcess"]["character_dict"])
    # The published medium model has 18,710 output classes: blank, the
    # configured dictionary, and the space token appended by CTCLabelDecode.
    chars.append(" ")
    if len(chars) + 1 != 18710:
        raise ValueError(f"Unexpected PP-OCRv6 CTC dictionary size: {len(chars) + 1}")
    return chars, {char: i + 1 for i, char in enumerate(chars)}


def encode_labels(texts: list[str], char_to_id: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for text in texts:
        try:
            rows.append([char_to_id[char] for char in text])
        except KeyError as exc:
            raise ValueError(f"RxHandBD label contains out-of-dictionary character {exc.args[0]!r}") from exc
    max_len = max(map(len, rows))
    labels = np.zeros((len(rows), max_len), dtype=np.int32)
    paddings = np.ones((len(rows), max_len), dtype=np.float32)
    for i, row in enumerate(rows):
        labels[i, :len(row)] = row
        paddings[i, :len(row)] = 0.0
    return labels, paddings


def image_batch(rows: list[dict[str, str]], *, augment: bool, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    batch = []
    for row in rows:
        with Image.open(row["image"]) as im:
            image = im.convert("RGB")
            if augment:
                image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.1))
                image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.9, 1.1))
            batch.append(resize_rec_image(image)[0])
    return np.stack(batch).astype(np.float32)


def batches(rows: list[dict[str, str]], batch_size: int, *, shuffle: bool, seed: int):
    order = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [rows[i] for i in order[start:start + batch_size]]


def ctc_inputs(texts: list[str], char_to_id: dict[str, int]):
    labels, label_paddings = encode_labels(texts, char_to_id)
    return labels, label_paddings


def build_step():
    @nnx.jit
    def step(model, optimizer, images, labels, label_paddings):
        def loss_fn(model):
            logits = model(images, training=True)
            logit_paddings = jnp.zeros(logits.shape[:2], dtype=jnp.float32)
            losses = optax.ctc_loss(
                logits.astype(jnp.float32), logit_paddings, labels,
                label_paddings, blank_id=0,
            )
            return jnp.mean(losses)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss
    return step


def decode_logits(logits: np.ndarray, characters: list[str]) -> list[str]:
    return ctc_greedy_decode(jnp.asarray(jax.nn.softmax(logits, axis=-1)), characters)


def evaluate(model, rows, char_to_id, characters, batch_size):
    total_loss = 0.0
    total_chars = 0
    total_edits = 0
    exact = 0
    samples = 0
    for batch_rows in batches(rows, batch_size, shuffle=False, seed=0):
        images = image_batch(batch_rows, augment=False, seed=0)
        texts = [r["text"] for r in batch_rows]
        labels, label_paddings = ctc_inputs(texts, char_to_id)
        logits = np.asarray(model(jnp.asarray(images), training=False))
        losses = np.asarray(optax.ctc_loss(
            jnp.asarray(logits), jnp.zeros(logits.shape[:2], dtype=jnp.float32),
            jnp.asarray(labels), jnp.asarray(label_paddings), blank_id=0,
        ))
        predictions = decode_logits(logits, characters)
        total_loss += float(losses.sum())
        for pred, truth in zip(predictions, texts):
            total_edits += edit_distance(pred, truth)
            total_chars += len(truth)
            exact += int(pred == truth)
            samples += 1
    return {
        "loss": total_loss / samples,
        "cer": total_edits / max(total_chars, 1),
        "exact_match": exact / max(samples, 1),
        "samples": samples,
    }


def run_training(args):
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()

    jax.config.update("jax_enable_x64", False)
    print(f"JAX backend: {jax.default_backend()}; devices: {jax.devices()}", flush=True)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model_yaml = args.model_dir / "inference.yml"
    characters, char_to_id = character_table(model_yaml)
    train_rows, val_rows = load_manifest(args.train_manifest), load_manifest(args.validation_manifest)
    valid_characters = set(char_to_id)
    validate_label_vocabulary(train_rows, valid_characters, split="train")
    validate_label_vocabulary(val_rows, valid_characters, split="validation")
    model = restore_nnx(args.checkpoint.resolve())
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.learning_rate, weight_decay=1e-5))
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    step_fn = build_step()

    metadata = {
        "base_checkpoint": str(args.checkpoint.resolve()),
        "dataset": "RxHandBD",
        "training_manifest": str(args.train_manifest.resolve()),
        "validation_manifest": str(args.validation_manifest.resolve()),
        "training_manifest_sha256": manifest_hash(args.train_manifest),
        "validation_manifest_sha256": manifest_hash(args.validation_manifest),
        "train_samples": len(train_rows),
        "validation_samples": len(val_rows),
        "objective": "CTC only; NRTR auxiliary head is unavailable in the inference checkpoint",
        "optimizer": "Optax AdamW; global norm clip 1.0",
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "early_stopping": {"metric": "validation CER", "patience": args.patience, "min_delta": args.min_delta},
        "seed": args.seed,
        "batch_norm_running_stats": "frozen to preserve fused-inference checkpoint semantics",
        "test_set_used": False,
        "backend": jax.default_backend(),
        "history": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing finetuned checkpoint: {args.output}")

    baseline = evaluate(model, val_rows, char_to_id, characters, args.batch_size)
    metadata["baseline_validation"] = baseline
    print(json.dumps({"baseline_validation": baseline}), flush=True)

    best_cer = baseline["cer"]
    patience_reference_cer = baseline["cer"]
    best_epoch = 0
    no_improvement_epochs = 0
    # Preserve the starting model as a candidate so this run never returns a
    # checkpoint worse than the continuation point.
    save_state(args.output, nnx.state(model))
    metadata["best_epoch"] = best_epoch
    metadata["best_validation"] = baseline
    metadata["best_checkpoint_source"] = str(args.checkpoint)
    (args.output / "finetuning.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        step_losses = []
        for batch_index, batch_rows in enumerate(batches(train_rows, args.batch_size, shuffle=True, seed=args.seed + epoch)):
            images = image_batch(batch_rows, augment=True, seed=args.seed + epoch * 100000 + batch_index)
            texts = [r["text"] for r in batch_rows]
            labels, label_paddings = ctc_inputs(texts, char_to_id)
            loss = float(step_fn(model, optimizer, jnp.asarray(images), jnp.asarray(labels), jnp.asarray(label_paddings)))
            step_losses.append(loss)
            if (batch_index + 1) % 25 == 0:
                print(json.dumps({"epoch": epoch + 1, "step": batch_index + 1, "steps": math.ceil(len(train_rows) / args.batch_size), "loss": loss}), flush=True)
        validation = evaluate(model, val_rows, char_to_id, characters, args.batch_size)
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(step_losses)),
            "validation": validation,
            "seconds": time.time() - epoch_start,
        }
        metadata["history"].append(record)
        print(json.dumps(record), flush=True)
        improved_best = validation["cer"] < best_cer
        significant_improvement = validation["cer"] < patience_reference_cer - args.min_delta
        if improved_best:
            best_cer = validation["cer"]
            best_epoch = epoch + 1
            # Always retain the numerically best validation CER.
            if args.output.exists():
                import shutil
                shutil.rmtree(args.output)
            save_state(args.output, nnx.state(model))
            metadata["best_epoch"] = best_epoch
            metadata["best_validation"] = validation
            metadata["best_checkpoint_source"] = "continued_training"
            (args.output / "finetuning.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if significant_improvement:
            no_improvement_epochs = 0
            patience_reference_cer = validation["cer"]
        else:
            no_improvement_epochs += 1
            record["epochs_without_improvement"] = no_improvement_epochs
            metadata["history"][-1] = record
        if no_improvement_epochs >= args.patience:
            metadata["stopped_early"] = True
            metadata["stop_reason"] = f"validation CER did not improve by {args.min_delta} for {args.patience} consecutive epochs"
            print(json.dumps({"early_stopping": metadata["stop_reason"], "best_epoch": best_epoch, "best_validation": metadata["best_validation"]}), flush=True)
            break

    metadata["total_seconds"] = time.time() - start_time
    (args.output / "finetuning.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_checkpoint": str(args.output.resolve()), "best_epoch": metadata.get("best_epoch"), "best_validation": metadata.get("best_validation"), "total_seconds": metadata["total_seconds"]}, indent=2), flush=True)
