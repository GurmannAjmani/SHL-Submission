# Grammar scoring from spoken audio

Predicts a grammar score between 1.0 and 5.0 for a spoken English answer. The
metric is mean absolute error (MAE). The notebook
`DebertaTrainingandcomparison.ipynb` runs the whole pipeline end to end and
writes `submission.csv`.

## Results from this run

Out-of-fold MAE across 732 training rows, 5-fold stratified cross-validation:

| Model | OOF MAE | RMSE | Pearson r | Fold MAE std |
|---|---|---|---|---|
| deberta_v3_large | 0.4489 | 0.5717 | 0.826 | 0.024 |
| deberta_v3_base | 0.4591 | 0.5811 | 0.820 | 0.028 |
| roberta_large | 0.4680 | 0.5886 | 0.814 | 0.013 |
| roberta_base | 0.4688 | 0.6007 | 0.807 | 0.025 |

Stacker comparison, repeated 5-fold CV, 3 repeats:

| Candidate | CV MAE |
|---|---|
| residual_gbm | 0.4285 |
| mean_of_models | 0.4424 |
| ridge_stack | 0.4434 |
| gbm_stack_with_features | 0.4439 |
| mae_weighted_blend | 0.4454 |
| best single model | 0.4489 |
| features_only_gbm | 0.5932 |

The submitted file uses `residual_gbm` with grid snapping, which scored 0.4092
on out-of-fold predictions.

## Why this configuration

DeBERTa-v3 large is the strongest single encoder here. Its disentangled
attention separates content from position, which suits a task where the same
words in a different order change the score. It leads every other encoder on
MAE, RMSE and correlation.

It is not submitted alone. The stacker averages all four encoders, then adds a
small gradient-boosted correction learned from the fluency features. That beats
DeBERTa-v3 large alone by 0.020 MAE. The four encoders make different mistakes,
so averaging cancels part of the error, and the fluency features carry
information no text model sees: speech rate, pause length, and Whisper's own
confidence in the transcript.

Grid snapping adds another 0.018. Every training label is a multiple of 0.5, so
a prediction of 3.47 is better reported as 3.5.

## Pipeline

1. **Transcription.** Whisper small converts audio to text. An initial prompt
   full of fillers biases it towards keeping disfluencies, which Whisper
   otherwise repairs. Repeats, false starts and fillers are grammar evidence.
   Per-segment log-probabilities, pause gaps and audio duration are cached with
   the transcript.
2. **Features.** Word and sentence counts, type-token ratio, filler and repeat
   counts, words per minute, pause ratio, and the Whisper confidence statistics.
   These feed the stacker, never the transformers.
3. **Fine-tuning.** Encoder, masked mean pooling, linear head. SmoothL1 loss on
   standardised labels, layer-wise learning-rate decay, cosine schedule with
   warmup, mixed precision, early stopping on validation MAE with the best
   epoch's weights restored.
4. **Stacking.** Seven candidates compete on repeated CV. The winner is refit on
   all training rows and applied to the test set.
5. **Submission.** Clip to the label range, snap to the 0.5 grid, write
   `submission.csv`. One single-model submission per encoder is written as a
   fallback.

## Key decisions

- **5-fold CV instead of a single holdout.** A 110-row holdout gave MAE anywhere
  from 0.43 to 0.77 across epochs of the same run. Model selection on that was
  guesswork.
- **Select epochs on MAE, not Pearson.** The highest-correlation epoch can carry
  a large bias. One DeBERTa base run peaked on Pearson at an epoch whose MAE was
  0.75.
- **SmoothL1 on standardised labels.** Closer to the metric than MSE.
  Standardising removes the first few epochs otherwise spent learning the scale.
- **Mean pooling, dropout disabled.** A stronger regression head than CLS
  pooling at this data size.
- **Dynamic padding to 512 tokens with gradient accumulation.** No truncation of
  long answers, and an effective batch of 16 rather than 4.

## Caveats

Out-of-fold MAE is slightly optimistic. Each fold's best epoch is chosen on that
fold's own validation rows, and the snapping decision was made on out-of-fold
predictions. Expect the leaderboard score to be somewhat higher than 0.4092.
Compare variants under this same protocol, and treat the leaderboard as the
final judge.

Training rows with a missing label, an empty transcript, or a label of 0 are
dropped. That leaves 732 of 769 rows.

## Improvements being tried

Ranked by expected gain per hour of compute.

1. **Transcript quality.** Run the pipeline with the disfluency prompt on and
   off and compare OOF MAE. Try Whisper medium. Transcription errors set a floor
   under everything downstream.
2. **More seeds.** `N_SEEDS = 3` averages away part of the fine-tuning variance.
   Fold predictions are cached, so this resumes after an interruption.
3. **Hyper-parameters.** `LOSS_BETA` from 0.1 to 0.5, per-model learning rates,
   dropout on or off, 8 to 15 epochs.
4. **More diverse encoders.** ELECTRA large or DeBERTa-v3 small. A blend gains
   more from a different architecture than from another seed of the same one.
5. **Pseudo-labelling.** Add confident test predictions as extra training rows,
   once the cross-validation setup is stable.

## Running it

The notebook expects the competition data at
`/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final` and writes
to `RUN_DIR`, which defaults to `/kaggle/working/model_comparison_v2`. A GPU is
required. The full run takes roughly 1 hour 50 minutes on a single accelerator.

Transcripts and per-fold predictions are cached, so a restarted session picks up
where it stopped. After changing a training setting, delete that model's folder
under `RUN_DIR`, or the stale cached predictions are reused.
