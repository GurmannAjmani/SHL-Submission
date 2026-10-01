# Grammar score from speech

This project gives a grammar score to a spoken English answer. The score is a
number from 1.0 to 5.0. The metric is the mean absolute error (MAE).

The notebook `DebertaTrainingandcomparison.ipynb` does the full task. It reads
the audio files and writes the submission files.

The text of this file follows ASD-STE100. The names of the models, the metrics
and the libraries are technical names. The text keeps them as they are.

## Files

| File | Contents |
|---|---|
| `DebertaTrainingandcomparison.ipynb` | The full pipeline, with the output of the run |
| `submission_deberta_v3_large.csv` | The predictions of the DeBERTa-v3 large encoder, for 216 test rows |
| `README.md` | This file |

## Approach

The task has 732 usable training rows. This is a small number for a transformer
model. The design therefore keeps the variance low at each step.

1. **Transcripts.** Whisper small changes each audio file into text. Whisper
   removes the fillers and repairs the repeats by default. These items show the
   grammar quality of the speaker. An initial prompt full of fillers therefore
   tells Whisper to keep them. The step also keeps the confidence of each
   segment, the length of each pause and the duration of the audio.
2. **Features.** The pipeline calculates the word count, the sentence length,
   the type-token ratio, the filler count, the repeat count, the speech rate and
   the pause ratio. Only the combination models read these features. The
   encoders read the transcript.
3. **Encoders.** Each encoder has a masked mean pooling layer and a linear head.
   The loss is SmoothL1 on the standardised labels. The optimiser uses a
   layer-wise learning-rate decay and a higher rate for the head. The schedule
   is a cosine decay with a warmup.
4. **Cross-validation.** Four stratified folds give an out-of-fold prediction
   for each training row. Each fold keeps the weights of the epoch with the
   lowest validation MAE. The test predictions of the four folds are averaged.
5. **Combination.** The two encoders and their combinations compete in a
   repeated 5-fold cross-validation. The candidate with the lowest MAE trains
   again on all the training rows.
6. **Submission.** The predictions stay in the label range. The pipeline then
   moves each prediction to the nearest label on the 0.5 grid.

## Results of this run

Two encoders, 4 folds, 6 epochs for each fold, 732 training rows.

| Encoder | Out-of-fold MAE | RMSE | Pearson r | Fold MAE std |
|---|---|---|---|---|
| deberta_v3_large | 0.4530 | 0.5809 | 0.821 | 0.028 |
| roberta_large | 0.4864 | 0.6170 | 0.797 | 0.037 |

The grid step lowers the MAE by a further 0.014.

DeBERTa-v3 large is the better of the two encoders. Its disentangled attention
keeps the content and the position apart. The same words in a different order
can thus give a different score. The encoder is better than RoBERTa large on all
three metrics.

## How the pipeline improved

The first version used one 110-row validation split, an MSE loss on the raw
labels, a CLS head with dropout, and one learning rate for all the layers. Its
validation MAE moved from 0.43 to 0.77 between the epochs of one run. These
changes made the result stable and lower:

| Change | Reason |
|---|---|
| Stratified cross-validation in place of one split | A 110-row split is too small. The model selection was not reliable. |
| Epoch selection on the MAE, not on the Pearson correlation | The metric is the MAE. An epoch with a high correlation can have a large bias. One run had an MAE of 0.75 at its best Pearson epoch. |
| SmoothL1 loss on the standardised labels | This loss is nearer to the metric than MSE. The standardised labels also remove the first epochs that only learn the scale. |
| Cosine decay across the epochs of the run | The first version planned a decay across 75 epochs but stopped near epoch 12. The learning rate thus stayed high. The last epochs now train at a small rate. |
| Mean pooling and no dropout | This head is better than CLS pooling for a regression task on a small dataset. |
| Layer-wise learning-rate decay and a separate head rate | The pretrained layers keep their knowledge. The new head learns more quickly. |
| Dynamic padding to 512 tokens and gradient accumulation | No long answer is cut. The effective batch is 16 and not 4. |
| A prompt that keeps the disfluencies | Whisper removes the fillers and the repeats by default. This removes the evidence of the fluency. |
| A combination of the encoders with a correction model | The two encoders make different errors. The mean removes part of the error. The correction model then uses the fluency features, which no encoder reads. |
| The 0.5 grid step | Each training label is a multiple of 0.5. A prediction of 3.47 is better as 3.5. |

The first version of this notebook also trained two base encoders. They were
less accurate than the two large encoders, so this version does not train them.
The compute now goes to more folds and more epochs for the large encoders.

## Possible improvements

1. **Transcript quality.** Make one run with the disfluency prompt and one run
   without it. Compare the out-of-fold MAE. Also try Whisper medium. A
   transcription error limits every later step.
2. **More encoders.** Add ELECTRA large or DeBERTa-v3 small. A different
   architecture helps a combination more than one more seed.
3. **Pseudo-labels.** Add the test rows with confident predictions to the
   training rows. Do this only when the cross-validation setup is stable.

## How to run the notebook

The notebook reads the data from
`/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final`. It writes
to `RUN_DIR`. A GPU is necessary.

The notebook caches the transcripts and the fold predictions. An interrupted
session thus continues from the last fold. A change to a training setting makes
the cached predictions incorrect. Remove the model folder in `RUN_DIR` before
you start the notebook again.
