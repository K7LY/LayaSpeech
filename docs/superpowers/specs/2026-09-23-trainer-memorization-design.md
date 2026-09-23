# LayaSpeech Trainer Memorization Diagnostic Design

## Intent

Before changing the Irodori teacher or increasing the corpus, establish whether the current LayaSpeech trainer can memorize its supervised labels. The decisive probe is duration-only training on 1, 4, and 10 utterances with no validation split. Teacher quality and dataset size are out of scope until that probe succeeds.

## Constraints

- Continue from branch `codex/irodori-poc` at `97f1de6270d16ea080230f228844b6de2211a8cc`.
- Reuse the local 30-utterance corpus and checkpoints when present.
- Run locally on CUDA; a silent CPU fallback is a failure.
- Do not add GitHub Actions.
- Do not track weights, WAV files, generated datasets, checkpoints, or large raw experiment output.
- Commit only generic trainer, evaluation, diagnostic, target, dataset, and documentation changes.
- Use held-out utterances disjoint from training utterances.

## Diagnostic sequence

1. Freeze the previous fine-tuned checkpoint's predictions and evaluate the same checkpoint on both train and held-out gold with one metric implementation.
2. Audit real encoded decisions end to end: state, qid, question type, criteria order, gold probabilities, numeric label, token markers, marker count, and option count.
3. Add generic trainer controls for one-control filtering, `ce` versus `rlcd_ce`, no-validation training, per-epoch train metrics, and fixed-step gradient/logit probes.
4. Run duration-only 1-, 4-, and 10-utterance overfit experiments. Compare CE-only and RLCD+CE on identical examples and seeds.
5. Classify the root cause from measured train accuracy, loss, gradients, logits, entropy, prediction distributions, and convergence. Fix production code only after a failing regression test identifies a concrete defect.
6. Only if tiny memorization succeeds, run full 30-sentence single-control experiments for pitch direction, duration, pitch level, and energy. Pause remains excluded.
7. Only after trainer correctness is established, test a boundary-only pause formulation and audit teacher F0/duration/energy extremes. Do not clamp or redesign targets without acoustic evidence.

## Interfaces

The generic training interface will retain `train_local(...)`. `TrainingConfig` will gain:

- `loss_mode`: `"rlcd_ce"` or `"ce"`.
- `control_names`: tuple of selected control suffixes, empty meaning all controls.
- `diagnostic_steps`: optimizer-step indices to snapshot.

Training history will report train loss/accuracy every epoch in addition to validation metrics. Diagnostics will contain total, encoder, and decision-head gradient norms; logits mean/std; softmax entropy; top-1 counts; and selected probe predictions at step 0 and requested optimizer steps.

Evaluation is label-index based at the model boundary and semantic-control based at policy output. This distinction prevents a rounded score result from hiding whether raw logits learned the correct class.

## Success criteria

- Encoding audit proves score target `[-3,+3]` maps bijectively to indices `[0,6]` and inference applies the inverse mapping.
- Tiny duration-only training reaches at least 95% training accuracy on one utterance and shows clear memorization on 4 and 10 utterances. A failure to reach this threshold blocks claims about teacher quality or data quantity.
- CE-only and RLCD+CE are compared using identical examples, initial checkpoint, seed, and optimizer settings.
- A causal classification is supported by measured evidence among trainer bug, RLCD/loss issue, multi-task interference, data shortage, feature/target design, or Laya suitability.
