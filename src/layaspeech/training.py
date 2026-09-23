"""Single-device Laya fine-tuning for the LayaSpeech PoC.

This intentionally mirrors Laya's published RLCD-style fine-tuning recipe while
remaining small enough to run locally on one GPU. It combines proper-scoring
policy-gradient guidance with ordinary soft cross-entropy.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 3
    micro_batch: int = 2
    grad_accum: int = 8
    group_size: int = 4
    encoder_lr: float = 2.5e-5
    head_lr: float = 1.0e-4
    sigma_start: float = 0.35
    sigma_end: float = 0.10
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    validation_fraction: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.micro_batch <= 0 or self.grad_accum <= 0 or self.group_size <= 1:
            raise ValueError("micro_batch/grad_accum must be > 0 and group_size > 1")
        if not 0.0 <= self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in [0, 0.5)")


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                rows.append(
                    {
                        "state": json.loads(row["state"]),
                        "questions": json.loads(row["questions"]),
                        "gold": json.loads(row["gold"]),
                    }
                )
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid training JSONL on line {line_number}") from exc
    if not rows:
        raise ValueError("training JSONL contains no rows")
    return rows


def _target_for_question(q: dict[str, Any], gold_q: dict[str, Any]) -> list[float]:
    qtype = q["type"]
    criteria = q.get("criteria", {})
    probabilities = gold_q.get("probabilities", {})

    if qtype == "choice":
        keys = list(criteria.keys())
        target = [float(probabilities.get(key, 0.0)) for key in keys]
    elif qtype == "noul":
        target = [
            float(probabilities.get("false", 0.5)),
            float(probabilities.get("true", 0.5)),
        ]
    elif qtype == "score":
        count = len(criteria) if isinstance(criteria, list) else 4
        target = [float(probabilities.get(str(i), 0.0)) for i in range(count)]
    else:
        raise ValueError(f"unsupported Laya question type: {qtype!r}")

    total = sum(target)
    if total <= 0:
        return [1.0 / len(target)] * len(target)
    return [value / total for value in target]


def _to_internal(q: dict[str, Any]) -> dict[str, Any]:
    qtype = q["type"]
    criteria = q.get("criteria")
    if qtype == "choice" and isinstance(criteria, list):
        criteria = {value: None for value in criteria}
    instructions = q["instructions"]
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False)
    return {"t": qtype, "ins": instructions, "crit": criteria}


def _build_items(rows: Iterable[dict[str, Any]], agent: Any) -> list[dict[str, Any]]:
    from laya.common import QTYPES, build_sequence, render_options

    cfg = agent.cfg
    items: list[dict[str, Any]] = []
    for row in rows:
        state = row["state"]
        questions = row["questions"]
        gold = row["gold"]
        for qid, q in questions.items():
            if qid not in gold:
                continue
            internal = _to_internal(q)
            target = _target_for_question(q, gold[qid])
            sequence, markers = build_sequence(
                agent.tok,
                state,
                internal,
                cfg.get("max_len", 1024),
                cfg.get("head_max_len", 256),
            )
            expected = len(render_options(internal))
            if len(markers) != expected:
                continue
            items.append(
                {
                    "ids": sequence,
                    "markers": markers,
                    "qtype": QTYPES[q["type"]],
                    "target": target,
                    "label": target.index(max(target)),
                    "qid": qid,
                }
            )
    if not items:
        raise ValueError("no valid Laya training items could be built")
    return items


def _device_from_string(torch: Any, requested: str | None) -> Any:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _save_checkpoint(agent: Any, output_dir: Path, metadata: dict[str, Any]) -> None:
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().half().contiguous().cpu()
        for key, value in agent.model.state_dict().items()
    }
    save_file(state, str(output_dir / "model.safetensors"))
    agent.model.encoder.config.save_pretrained(output_dir / "encoder")
    agent.tok.save_pretrained(output_dir / "tokenizer")

    cfg = dict(agent.cfg)
    cfg["fine_tuned"] = True
    cfg["model_name"] = "layaspeech-prosody"
    cfg["temperature"] = [1.0, 1.0, 1.0]
    cfg["temperature_by_options"] = {}
    (output_dir / "rl_agent_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "training_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _evaluate(agent: Any, items: list[dict[str, Any]], device: Any, batch_size: int) -> dict[str, float]:
    import torch
    from laya.common import collate_items

    if not items:
        return {"loss": float("nan"), "accuracy": float("nan")}

    model = agent.model
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    with torch.no_grad():
        for offset in range(0, len(items), batch_size):
            chunk = items[offset : offset + batch_size]
            batch = collate_items([[item] for item in chunk], agent.tok.pad_token_id)
            logits, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )
            mask = batch["marker_mask"].to(device)
            target = batch["target"].to(device)
            logits = logits.float().masked_fill(~mask, -1e4)
            loss = -(target * torch.log_softmax(logits, -1)).sum(-1)
            losses.extend(float(value) for value in loss.cpu())
            prediction = logits.argmax(-1).cpu()
            labels = batch["label"]
            correct += int((prediction == labels).sum())
            total += len(chunk)
    model.train()
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "accuracy": correct / max(1, total),
    }


def train_local(
    *,
    training_jsonl: str | Path,
    output_dir: str | Path,
    model_id: str = "convaiinnovations/laya",
    subfolder: str | None = "multilingual",
    device: str | None = None,
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    """Fine-tune Laya on one device and save a normal Laya checkpoint."""

    config = config or TrainingConfig()

    try:
        import laya
        import torch
        from laya.common import collate_items, proper_reward
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are not installed. Install with: uv sync --extra train"
        ) from exc

    resolved_device = _device_from_string(torch, device)
    agent = laya.load(model_id, device=str(resolved_device), subfolder=subfolder)
    if agent.device != resolved_device:
        raise RuntimeError(
            f"Laya fell back to {agent.device}; requested training device was {resolved_device}"
        )
    model = agent.model
    model.train()

    try:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.head_checkpointing = True
    except (AttributeError, TypeError):
        pass

    rows = _load_rows(training_jsonl)
    rng = random.Random(config.seed)
    rng.shuffle(rows)

    val_count = (
        max(1, round(len(rows) * config.validation_fraction))
        if len(rows) > 1 and config.validation_fraction > 0
        else 0
    )
    validation_rows = rows[:val_count]
    training_rows = rows[val_count:]
    if not training_rows:
        training_rows = validation_rows
        validation_rows = []

    train_items = _build_items(training_rows, agent)
    val_items = _build_items(validation_rows, agent) if validation_rows else []

    encoder_params = [
        parameter
        for name, parameter in model.named_parameters()
        if "encoder." in name and parameter.requires_grad
    ]
    head_params = [
        parameter
        for name, parameter in model.named_parameters()
        if "encoder." not in name and parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": config.encoder_lr},
            {"params": head_params, "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )

    updates_per_epoch = max(
        1,
        math.ceil(len(train_items) / (config.micro_batch * config.grad_accum)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, updates_per_epoch * config.epochs),
        eta_min=1e-6,
    )

    use_amp = resolved_device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(config.epochs):
        rng.shuffle(train_items)
        progress = epoch / max(1, config.epochs - 1)
        sigma = config.sigma_start + (config.sigma_end - config.sigma_start) * progress
        epoch_losses: list[float] = []
        for accumulation, offset in enumerate(
            range(0, len(train_items), config.micro_batch), start=1
        ):
            chunk = train_items[offset : offset + config.micro_batch]
            batch = collate_items([[item] for item in chunk], agent.tok.pad_token_id)
            context = (
                torch.autocast("cuda", dtype=amp_dtype)
                if use_amp
                else nullcontext()
            )

            with context:
                logits, act = model(
                    batch["input_ids"].to(resolved_device),
                    batch["attention_mask"].to(resolved_device),
                    batch["marker_pos"].to(resolved_device),
                    batch["marker_mask"].to(resolved_device),
                    batch["qtype"].to(resolved_device),
                )

            logits = logits.float()
            mask = batch["marker_mask"].to(resolved_device)
            target = batch["target"].to(resolved_device)
            qtype = batch["qtype"].to(resolved_device)
            option_count = mask.sum(-1, keepdim=True).float()

            epsilon = (
                torch.randn(
                    (config.group_size,) + logits.shape,
                    device=resolved_device,
                )
                * sigma
                * mask
            )
            epsilon = (
                epsilon - epsilon.sum(-1, keepdim=True) / option_count
            ) * mask
            sampled_logits = logits.detach().unsqueeze(0) + epsilon
            sampled_probs = torch.softmax(
                sampled_logits.masked_fill(~mask, -1e4),
                -1,
            )

            with torch.no_grad():
                reward = proper_reward(
                    sampled_probs,
                    target.unsqueeze(0),
                    qtype,
                    mask,
                    w_sph=0.75,
                    w_rps=1.0,
                )
                advantage = reward - reward.mean(0, keepdim=True)
                advantage = advantage / (advantage.std() + 1e-6)

            log_probability = -(
                ((sampled_logits - logits.unsqueeze(0)) ** 2) * mask
            ).sum(-1) / (2 * sigma**2)
            loss_rl = -(advantage * log_probability).mean()
            loss_ce = -(
                target
                * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
            ).sum(-1).mean()
            loss = (loss_rl + loss_ce) / config.grad_accum + 0.0 * act.sum()

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            last_batch = offset + config.micro_batch >= len(train_items)
            if accumulation % config.grad_accum == 0 or last_batch:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_losses.append(float(loss.detach().cpu()) * config.grad_accum)

        metrics = _evaluate(
            agent,
            val_items,
            resolved_device,
            max(1, config.micro_batch),
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": sum(epoch_losses) / max(1, len(epoch_losses)),
            "validation_loss": metrics["loss"],
            "validation_accuracy": metrics["accuracy"],
            "sigma": sigma,
        }
        history.append(record)

        _save_checkpoint(
            agent,
            Path(output_dir) / "checkpoint_latest",
            {
                "config": asdict(config),
                "model_id": model_id,
                "subfolder": subfolder,
                "device": str(resolved_device),
                "train_items": len(train_items),
                "validation_items": len(val_items),
                "history": history,
            },
        )

    final_meta = {
        "config": asdict(config),
        "model_id": model_id,
        "subfolder": subfolder,
        "device": str(resolved_device),
        "train_items": len(train_items),
        "validation_items": len(val_items),
        "history": history,
    }
    _save_checkpoint(agent, Path(output_dir), final_meta)
    return final_meta
