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

CONTROL_NAMES = (
    "pitch_level",
    "pitch_direction",
    "duration_level",
    "energy_level",
    "pause_after",
)
LOSS_MODES = ("ce", "rlcd_ce")


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
    loss_mode: str = "rlcd_ce"
    control_names: tuple[str, ...] = ()
    diagnostic_steps: tuple[int, ...] = (0, 1, 10, 100)

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.micro_batch <= 0 or self.grad_accum <= 0 or self.group_size <= 1:
            raise ValueError("micro_batch/grad_accum must be > 0 and group_size > 1")
        if not 0.0 <= self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in [0, 0.5)")
        if self.loss_mode not in LOSS_MODES:
            raise ValueError(f"loss_mode must be one of {LOSS_MODES!r}")
        invalid_controls = set(self.control_names) - set(CONTROL_NAMES)
        if invalid_controls:
            raise ValueError(f"unsupported control names: {sorted(invalid_controls)!r}")
        if any(step < 0 for step in self.diagnostic_steps):
            raise ValueError("diagnostic steps must be >= 0")


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


def _build_items(
    rows: Iterable[dict[str, Any]],
    agent: Any,
    control_names: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    from laya.common import QTYPES, build_sequence, render_options

    cfg = agent.cfg
    items: list[dict[str, Any]] = []
    for row in rows:
        state = row["state"]
        questions = row["questions"]
        gold = row["gold"]
        for qid, q in questions.items():
            if control_names and not any(qid.endswith(f"_{name}") for name in control_names):
                continue
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
                raise ValueError(
                    f"question {qid!r} marker/option mismatch: "
                    f"got {len(markers)}, expected {expected}"
                )
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


def _supervised_cross_entropy(logits: Any, target: Any, mask: Any) -> Any:
    import torch

    masked_logits = logits.float().masked_fill(~mask, -1e4)
    return -(target * torch.log_softmax(masked_logits, -1)).sum(-1).mean()


def combine_training_loss(loss_ce: Any, loss_rlcd: Any, loss_mode: str) -> Any:
    """Combine loss terms without letting RLCD affect a CE-only experiment."""

    if loss_mode == "ce":
        return loss_ce
    if loss_mode == "rlcd_ce":
        return loss_ce + loss_rlcd
    raise ValueError(f"unsupported loss_mode: {loss_mode!r}")


def _parameter_gradient_norm(parameters: Iterable[Any]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(total)


def _diagnostic_snapshot(model: Any, logits: Any, mask: Any) -> dict[str, Any]:
    import torch

    named_parameters = list(model.named_parameters())
    encoder_parameters = [
        parameter
        for name, parameter in named_parameters
        if name.startswith("encoder.") or ".encoder." in name
    ]
    head_parameters = [
        parameter
        for name, parameter in named_parameters
        if not (name.startswith("encoder.") or ".encoder." in name)
    ]
    valid_logits = logits.float().masked_select(mask)
    masked_logits = logits.float().masked_fill(~mask, -1e4)
    probabilities = torch.softmax(masked_logits, -1) * mask
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-9))).sum(-1)
    return {
        "gradient_norm": _parameter_gradient_norm(parameter for _, parameter in named_parameters),
        "encoder_gradient_norm": _parameter_gradient_norm(encoder_parameters),
        "decision_head_gradient_norm": _parameter_gradient_norm(head_parameters),
        "logits_mean": float(valid_logits.mean().detach().cpu()),
        "logits_std": float(valid_logits.std(unbiased=False).detach().cpu()),
        "softmax_entropy": float(entropy.mean().detach().cpu()),
        "predicted_top1_class": masked_logits.argmax(-1).detach().cpu().tolist(),
    }


def _capture_diagnostic(
    agent: Any,
    items: list[dict[str, Any]],
    device: Any,
) -> dict[str, Any]:
    import torch
    from laya.common import collate_items

    batch = collate_items([[item] for item in items], agent.tok.pad_token_id)
    model = agent.model
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits, _ = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["marker_pos"].to(device),
            batch["marker_mask"].to(device),
            batch["qtype"].to(device),
        )
    snapshot = _diagnostic_snapshot(model, logits, batch["marker_mask"].to(device))
    snapshot["qids"] = [item["qid"] for item in items]
    snapshot["gold_labels"] = [item["label"] for item in items]
    model.train(was_training)
    return snapshot


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

    train_items = _build_items(training_rows, agent, config.control_names)
    val_items = (
        _build_items(validation_rows, agent, config.control_names) if validation_rows else []
    )

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
    diagnostic_items = train_items[: max(1, config.micro_batch)]
    diagnostic_steps = set(config.diagnostic_steps)
    diagnostic_snapshots: list[dict[str, Any]] = []
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    if 0 in diagnostic_steps:
        diagnostic_snapshots.append(
            {"step": 0, **_capture_diagnostic(agent, diagnostic_items, resolved_device)}
        )

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

            loss_ce = _supervised_cross_entropy(logits, target, mask)
            loss_rl = logits.new_zeros(())
            if config.loss_mode == "rlcd_ce":
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
            loss = (
                combine_training_loss(loss_ce, loss_rl, config.loss_mode)
                / config.grad_accum
                + 0.0 * act.sum()
            )

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
                global_step += 1
                if global_step in diagnostic_steps:
                    diagnostic_snapshots.append(
                        {
                            "step": global_step,
                            **_capture_diagnostic(agent, diagnostic_items, resolved_device),
                        }
                    )
                optimizer.zero_grad(set_to_none=True)

            epoch_losses.append(float(loss.detach().cpu()) * config.grad_accum)

        train_metrics = _evaluate(
            agent,
            train_items,
            resolved_device,
            max(1, config.micro_batch),
        )
        validation_metrics = _evaluate(
            agent,
            val_items,
            resolved_device,
            max(1, config.micro_batch),
        )
        record = {
            "epoch": epoch + 1,
            "optimization_loss": sum(epoch_losses) / max(1, len(epoch_losses)),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
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
                "diagnostic_snapshots": diagnostic_snapshots,
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
        "diagnostic_snapshots": diagnostic_snapshots,
    }
    _save_checkpoint(agent, Path(output_dir), final_meta)
    return final_meta
