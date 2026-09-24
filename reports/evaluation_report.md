# HandsToVoice — Sign Recognition Evaluation

Generated 2026-09-24 11:34 by `evaluate_model.py`.

## Method

- **428 recorded sequences** of **36 signs** (7–17 per sign).
- **5-fold stratified cross-validation**: each sequence is predicted by a model trained without it, using the app's own training pipeline (`src/training.py`: same augmentation, architecture and early stopping).
- **Face-missing test**: the same held-out sequences with the face-position features removed, as happens live when the face detector misses.
- **Latency**: time for one prediction by the deployed model, as the app calls it.

## Results

| Measure | Result |
|---|---|
| Accuracy (held-out) | **82.7%** |
| Macro-average F1 | 0.825 |
| Accuracy with face not detected | 80.4% |
| Prediction latency | 4.3 ms mean, 5.7 ms 95th percentile |

Confusion matrix: `confusion_matrix.png`.

## Per-sign results

| Sign | Precision | Recall | F1 | Samples |
|---|---|---|---|---|
| itonde | 0.67 | 0.44 | 0.53 | 9 |
| mbabarira | 0.67 | 0.50 | 0.57 | 12 |
| kuwa_gatanu | 0.75 | 0.50 | 0.60 | 12 |
| benshi | 0.53 | 0.73 | 0.62 | 11 |
| gusangiza | 0.62 | 0.62 | 0.62 | 13 |
| kuwa_kane | 0.60 | 0.75 | 0.67 | 12 |
| mama | 0.73 | 0.67 | 0.70 | 12 |
| masenge | 0.62 | 0.80 | 0.70 | 10 |
| abana | 0.64 | 0.82 | 0.72 | 11 |
| tangira | 0.73 | 0.73 | 0.73 | 11 |
| kuwa_gatatu | 0.69 | 0.82 | 0.75 | 11 |
| oya | 1.00 | 0.62 | 0.77 | 16 |
| umunsi | 0.70 | 0.88 | 0.78 | 8 |
| kucyumweru | 0.77 | 0.83 | 0.80 | 12 |
| kuwa_gatandatu | 0.83 | 0.77 | 0.80 | 13 |
| yego | 0.75 | 0.86 | 0.80 | 14 |
| mwiza | 0.85 | 0.79 | 0.81 | 14 |
| kuwa_kabiri | 1.00 | 0.73 | 0.84 | 11 |
| atandukanye | 0.79 | 0.92 | 0.85 | 12 |
| ngewe | 0.82 | 0.90 | 0.86 | 10 |
| witeguye | 0.80 | 0.92 | 0.86 | 13 |
| kuwa_mbere | 0.92 | 0.85 | 0.88 | 13 |
| amazina_yange | 0.92 | 0.86 | 0.89 | 14 |
| wowe | 0.87 | 0.93 | 0.90 | 14 |
| birasobanutse | 0.91 | 0.91 | 0.91 | 11 |
| muraho | 0.88 | 1.00 | 0.93 | 7 |
| umeze_gute_?? | 1.00 | 0.88 | 0.93 | 16 |
| uyu_munsi | 1.00 | 0.92 | 0.96 | 13 |
| neza | 1.00 | 0.93 | 0.96 | 14 |
| tugiye | 0.94 | 1.00 | 0.97 | 17 |
| ameze | 1.00 | 1.00 | 1.00 | 7 |
| marume | 1.00 | 1.00 | 1.00 | 11 |
| ndagukunda | 1.00 | 1.00 | 1.00 | 10 |
| ni | 1.00 | 1.00 | 1.00 | 10 |
| nyabuneka | 1.00 | 1.00 | 1.00 | 11 |
| papa | 1.00 | 1.00 | 1.00 | 13 |

## Most confused signs

| Actual | Predicted as | Times |
|---|---|---|
| mbabarira | masenge | 5 |
| tangira | atandukanye | 2 |
| oya | yego | 2 |
| oya | witeguye | 2 |
| mwiza | gusangiza | 2 |
| masenge | mbabarira | 2 |
| mama | kucyumweru | 2 |
| kuwa_kane | ngewe | 2 |
| kuwa_gatanu | benshi | 2 |
| kuwa_gatandatu | abana | 2 |

## Limitations

- **All recordings are from one signer**, so these numbers measure how well the model recognizes new attempts by the *same* person. Accuracy on a different signer is not measured here and is expected to be lower; that requires recordings from additional signers, kept entirely out of training.
- Signs are recognized one at a time (isolated-sign recognition), not as continuous signed sentences.
- Recordings were made in a limited set of rooms and lighting conditions.
