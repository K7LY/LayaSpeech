# Trainer Memorization Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine with causal evidence whether LayaSpeech's trainer can memorize supervised prosody labels, and fix only verified generic defects.

**Architecture:** Preserve the previous corpus and checkpoint as immutable evidence. Add a reusable label-index evaluator and narrowly configurable training diagnostics, then run gated duration-only tiny-overfit experiments before any teacher, pause, or target redesign.

**Tech Stack:** Python 3.12, PyTorch 2.10 CUDA 12.8, Laya 0.3.6, pytest, Ruff, JSONL experiment artifacts.

**Spec:** `docs/superpowers/specs/2026-09-23-trainer-memorization-design.md`

## Global Constraints

- Work on `codex/irodori-poc`; preserve local `.poc-local` artifacts.
- CUDA is mandatory for training experiments.
- Never add GitHub Actions.
- Never track weights, WAVs, generated datasets, checkpoints, or large raw evaluation output.
- Do not investigate teacher-data quantity until tiny memorization is decided.
- Production changes use RED-GREEN TDD and remain provider-neutral.

## Review Focus

- Score targets at -3, 0, and +3 must encode to labels 0, 3, and 6 and decode back exactly.
- Choice criteria order must be stable between dataset export, training item construction, and inference.
- A control filter must select exact qid suffixes without accidental substring matches or empty silent datasets.
- CE-only must not execute RLCD sampling/reward code, and RLCD+CE must retain current behavior.
- Gradient accumulation must produce an optimizer step for the final partial group and diagnostics must refer to optimizer steps, not micro-batches.

---

### Task 1: Freeze and reproduce original checkpoint metrics

**Files:**
- Local-only: `.poc-local/trainer-audit/original-{train,heldout}-predictions.json`
- Local-only: `.poc-local/trainer-audit/original-metrics.json`
- Create: `src/layaspeech/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: semantic mora records with the five control fields.
- Produces: `evaluate_control_records(gold, predictions, majority_source) -> dict` with accuracy, majority baseline, prediction distribution, confusion matrix, and ordinal MAE per control.

- [ ] Write tests using two tiny records that pin accuracy, class counts, confusion orientation, majority values, and ordinal MAE.
- [ ] Run `pytest tests/test_evaluation.py -q` and verify failure because the module does not exist.
- [ ] Implement the metric function without torch or Laya dependencies.
- [ ] Run the focused test and full local check.
- [ ] Evaluate the preserved checkpoint against all 24 train and 6 held-out utterances using identical inference and save local JSON evidence.
- [ ] Commit the generic evaluator and tests.

### Task 2: Audit dataset encoding and inference mapping

**Files:**
- Create: `src/layaspeech/diagnostics.py`
- Test: `tests/test_diagnostics.py`
- Local-only: `.poc-local/trainer-audit/encoding-audit.json`

**Interfaces:**
- Consumes: decoded JSONL rows plus a loaded Laya agent.
- Produces: `audit_training_rows(rows, agent, limit) -> list[dict]` containing state, qid, qtype, criteria, gold probabilities, label, ids, markers, marker count, and expected option count.

- [ ] Write a failing test proving score levels -3/0/+3 become indices 0/3/6 and `score_index_to_level` reverses them.
- [ ] Write a failing test proving choice criteria order and qid uniqueness are preserved.
- [ ] Implement explicit `level_to_score_index` and `score_index_to_level` helpers and use them in dataset export and policy parsing.
- [ ] Implement row auditing with a hard error on marker/option mismatch; do not silently skip malformed decisions.
- [ ] Run focused tests and the complete local check.
- [ ] Dump representative real decisions for all control types and inspect them manually.
- [ ] Commit encoding helpers, audit tool, and tests.

### Task 3: Add controlled loss modes and training instrumentation

**Files:**
- Modify: `src/layaspeech/training.py`
- Modify: `src/layaspeech/cli.py`
- Test: `tests/test_training.py`
- Test: `tests/test_cli_training.py`

**Interfaces:**
- Extends `TrainingConfig` with `loss_mode`, `control_names`, and `diagnostic_steps`.
- Extends `_build_items(rows, agent, control_names=())` with exact qid-suffix filtering.
- Training metadata gains per-epoch train metrics and fixed-step diagnostic snapshots.

- [ ] Write failing config tests for accepted/rejected loss modes and control names.
- [ ] Write a failing item-filter test that selects only `_duration_level` and rejects an empty result.
- [ ] Extract a tested supervised cross-entropy helper and a tested `combine_training_loss` helper whose CE mode ignores RLCD loss.
- [ ] Add gradient/logit diagnostic calculation with unit tests over a tiny torch module.
- [ ] Record train loss and accuracy every epoch using `_evaluate` on train items.
- [ ] Expose `--loss-mode`, repeatable `--control`, and `--diagnostic-step` CLI options.
- [ ] Run focused tests and the full local check.
- [ ] Commit configurable trainer instrumentation.

### Task 4: Run duration-only tiny overfit and CE/RLCD comparison

**Files:**
- Local-only: `.poc-local/trainer-audit/tiny-inputs/`
- Local-only: `.poc-local/trainer-audit/tiny-checkpoints/`
- Local-only: `.poc-local/trainer-audit/tiny-results.json`

**Interfaces:**
- Consumes: Task 3 trainer controls and fixed utterance IDs from the previous train split.
- Produces: 1-, 4-, and 10-utterance convergence results for CE and RLCD+CE.

- [ ] Materialize deterministic duration-only inputs for IDs `u001`; `u001,u002,u003,u004`; and `u001` through `u011` excluding held-out IDs.
- [ ] Verify every row contains only duration qids and that all selected labels are present in the source gold.
- [ ] Run CE-only with validation fraction 0, seed 20260923, and enough epochs up to 100 to reach or disprove 95% train accuracy.
- [ ] Run RLCD+CE on identical inputs/settings.
- [ ] Record loss, raw-item train accuracy, semantic policy accuracy, predicted distributions, convergence epoch, and step 0/1/10/100 diagnostics.
- [ ] Classify tiny memorization success before continuing.

### Task 5: Fix only a demonstrated trainer defect

**Files:**
- Modify only the generic module implicated by Task 4 evidence.
- Test: the smallest existing or new test module covering the defect.

**Interfaces:**
- Consumes: Task 4 causal evidence.
- Produces: a regression test that fails on `97f1de6` behavior and a minimal generic fix.

- [ ] State one root-cause hypothesis linked to a measured boundary.
- [ ] Write and run the minimal failing regression test.
- [ ] Implement one fix and re-run the test.
- [ ] Re-run the 1-utterance duration probe and require at least 95% train accuracy.
- [ ] Run the full local check and commit the fix.

### Task 6: Gate downstream experiments on memorization

**Files:**
- Local-only: `.poc-local/trainer-audit/control-results.json`
- Local-only: `.poc-local/trainer-audit/pause-results.json`
- Local-only: `.poc-local/trainer-audit/extreme-audit.json`
- Modify production target/analysis code only after a separate RED-GREEN defect reproduction.

**Interfaces:**
- Consumes: successful Task 4/5 tiny memorization.
- Produces: single-control full-corpus results, optional boundary-only pause results, and acoustic extreme evidence.

- [ ] If tiny memorization fails after the verified trainer path, stop and classify without blaming teacher/data volume.
- [ ] If it succeeds, train pitch direction, duration, pitch level, and energy separately on the 24/6 split and report train/held-out metrics.
- [ ] Compare single-control outcomes against the original five-control model to test multi-task interference.
- [ ] Only then test boundary detection followed by non-none pause length on boundary candidates.
- [ ] Audit top/bottom 20 pitch extremes plus duration/energy extremes with WAV intervals, MFA boundaries, contours, and raw WORLD values.
- [ ] Change extraction/targets only if the audit demonstrates a reproducible generic defect.

### Task 7: Verify and report causal classification

**Files:**
- Create: `docs/TRAINER_DIAGNOSTICS.md`
- User deliverable: `outputs/LayaSpeech-trainer-memorization-report.md`

**Interfaces:**
- Consumes: all prior metric and diagnostic JSON.
- Produces: evidence-backed classification among trainer bug, RLCD/loss issue, multi-task interference, data shortage, feature/target design, and Laya suitability.

- [ ] Run `scripts/check.ps1`, compile, Ruff, git diff check, artifact leak scan, and verify GitHub Actions remain absent.
- [ ] Record original train/held-out results, tiny 1/4/10 results, CE/RLCD comparison, encoding audit, gradient/logit audit, control experiments, pause/F0 gates, root cause, changes, tests, and commit SHA.
- [ ] Commit only generic code and documentation.
