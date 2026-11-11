# Social Media Audio Cut and AI-Dubbing Detector

A PyTorch research pipeline that flags two kinds of audio manipulation in 30-second social media clips:

- **cut**: non-AI audio spliced in from another source
- **dub**: part or all of the clip is AI-generated (dubbed) voice

The two binary outputs are combined into four classes: `real`, `cut`, `dubbed`, `cut_and_dubbed`. Music-only clips are excluded.

> **Research use only.** This project is an academic experiment. It is not a forensic tool and must not be used to accuse a person or a creator, to make moderation, legal, employment or other consequential decisions, or to claim that a given video is authentic or fake. The model makes errors in both directions, was trained on a small dataset of Indonesian-language social media clips, and has not been validated outside that domain.

## Links

| Resource | Link |
|---|---|
| Notebook | `AI_Social_Media_Dubbing_Detector_Final.ipynb` (designed for Google Colab) |
| Spectrogram dataset (Kaggle) | https://www.kaggle.com/datasets/dherylc/spectrogram-ai-social-media-short-video-detector |
| Labeling sheet (Google Sheets) | https://docs.google.com/spreadsheets/d/1zI5Zj_G64kvpQk-ZMS6TzfNQPj9XNyP960X-lkjWcIM |

Access to the sheet and to any private Kaggle dataset is controlled by their owner. If you cannot open them, request access or point the notebook at your own copies (see "Changing the dataset").

## How it works

### Pipeline

1. **Sync.** The notebook lists the `.npz` files in the Kaggle dataset and appends any filename that is not yet in the Google Sheet, with the label `Unclassified`. Existing labels are never overwritten.
2. **Load.** Labels come from the sheet. Rows still marked `Unclassified`, and rows whose label is in `EXCLUDE_LABELS` (`music`), are skipped. All spectrograms are normalized and cached in RAM (about 1.5 GB for roughly 1,500 clips). If a file is missing, training stops with an error. There is no synthetic fallback.
3. **Audit and groups.** Filenames follow `<handle><video id><tags>_NNN.npz`, where `_001`, `_002`, ... are consecutive 30s segments of one video and `<tags>` appears to list the per-segment labels in order (for example `cutdubreal`: segment 1 cut+dub, segment 2 real). The notebook checks that against the sheet and prints the agreement rate and any disagreements. The filename minus `_NNN.npz` is the **group key**, so all segments of a video stay in the same fold. The filename is never fed to the model.
4. **Features.** Computed on the GPU from the pad-aware normalized log-Mel `(80, 3001)`:
   - channel 0, static: per-frequency-bin z-score over active frames
   - channel 1, delta: frame-to-frame change
   - channel 2, splice-step: mean of the next 25 frames minus mean of the previous 25, which highlights a sustained change in noise floor or room tone as caused by a splice

   Padding stays exactly zero in every channel.
5. **Model.** ResNet-18 with ImageNet weights, max and mean pooling concatenated, then `Linear(1024,256) -> BatchNorm -> ReLU -> Dropout(0.5) -> Linear(256,2)`. The two outputs are independent sigmoids (cut, dub). Loss is `BCEWithLogitsLoss` with a square-root smoothed `WeightedRandomSampler` for class balance. Augmentation is noise on active frames plus one frequency mask. There is no time masking, because it can erase the short edit that carries the label.
6. **Evaluation.** Stratified, grouped 5-fold cross-validation. Each fold trains from scratch for a fixed number of epochs and produces out-of-fold probabilities. The report gives per-head average precision, F1-optimal thresholds, a derived four-class report and confusion matrix, the false-alarm rate on `real` clips, and the recall on manipulated clips. Thresholds are tuned on the same out-of-fold predictions, so tuned scores are slightly optimistic.
7. **Final model.** Trained on all labeled data and saved to `audio_deepfake_resnet18_weights.pt` with class order, thresholds, feature settings and preprocessing settings.
8. **Inference.** Audio or video is resampled to 16 kHz mono and cut into 30s chunks (a trailing chunk under 5s is dropped). Each chunk becomes a log-Mel spectrogram, is zero-padded in the spectrogram domain to 3001 frames, normalized with the same function used in training, and scored. Output is per segment: `p(cut)`, `p(dub)` and a derived label.
9. **Robustness test.** Optional cell that scores a labeled external set as-is and after MP3/AAC re-encoding at several bitrates.
10. **Embedding probe (optional).** WavLM embeddings from the raw MP4 dataset, probed with logistic regression on the same folds, to compare against the CNN.

### Notebook sections

| Section | Purpose |
|---|---|
| 1-3 | Imports, Google and Kaggle authentication, sheet sync |
| 4 Configuration | All tunable settings in one cell |
| 5 Data | Indexing, label audit, groups, RAM cache |
| 6 Features | Feature builder and augmentation, with a self-check |
| 7 Model | Model, sampler, training and prediction helpers |
| 8 Cross-validation | Grouped CV, out-of-fold report, threshold tuning |
| 9 Final model | Train on all data and export the checkpoint |
| 10 Inference | Inference functions and the preprocessing parity check |
| 11 Uploads | Score a file you upload |
| 12 External set | Labeled external clips and the MP3/AAC codec test |
| 13 Optional | WavLM embedding probe |

## Requirements

- Google Colab with a GPU runtime (a T4 is enough). The notebook can also run inside a Kaggle notebook, where the dataset is read from `/kaggle/input/` and no download is needed.
- Python packages: `torch`, `torchvision`, `librosa`, `scikit-learn`, `pandas`, `numpy`, `tqdm`, `gspread`, `kaggle`. For the optional probe also `transformers`.
- `ffmpeg` on the path for the codec test and for reading `.mp4` uploads (preinstalled on Colab).
- Accounts: a Google account with access to the labeling sheet, and a Kaggle account with an API token.

## Changing the dataset

### Accounts and credentials

- **Google.** The notebook calls `auth.authenticate_user()` in Colab, so the Google account you sign in with must have at least edit access to the sheet (the sync step appends rows). Share the sheet with that account, or use your own copy.
- **Kaggle.** Create a token at Kaggle, Settings, API, "Create New Token". Upload the resulting `kaggle.json` when the notebook asks. It is copied to `~/.kaggle/`. Do not commit this file or share it.

### Pointing at your own sheet and dataset

In the sync cell (section 3), change the two constants:

```python
SPREADSHEET_ID = '<the long id in your sheet URL: docs.google.com/spreadsheets/d/<ID>/edit>'
DATASET_NAME   = '<kaggle-owner>/<dataset-slug>'
```

The data cell (section 5) reuses `DATASET_NAME`. If the dataset is not found under `/kaggle/input/<slug>`, it downloads it to `/content/npz_features`.

### Sheet format

The first worksheet is read. It needs a header row with these columns:

| Column | Content |
|---|---|
| `filename` (or `file_name`) | the `.npz` filename. Folder prefixes are stripped, and `.npz` is added if missing |
| `manual_classification` (or `classification`) | one of `real`, `cut`, `dubbed`, `cut_and_dubbed`, `music`, or `Unclassified` |

The sync step appends new rows as `[filename, 'Unclassified']`, so keep `filename` in column A and the label in column B. Labels are trimmed and lowercased on load. Any other label value stops the notebook with an error that shows the labels it found.

### `.npz` file format

Each file holds a single log-Mel array. The notebook reads the first array in the archive, whatever its key. Requirements:

- shape `(80, T)` with `T <= 3001`; shorter clips are zero-padded at the end of the time axis, longer ones are cropped
- 16 kHz audio, `n_fft=400`, `win_length=400`, `hop_length=160`, 80 mel bins, `fmin=0`, `fmax=8000`, power 2.0
- values in decibels from `librosa.power_to_db(..., ref=np.max)` (about -80 to 0)
- tail padding is exact zeros in the spectrogram domain, added after the dB conversion

To build a compatible file from a 30s audio clip:

```python
import numpy as np, librosa
y, _ = librosa.load("clip.wav", sr=16000, mono=True)
mel = librosa.feature.melspectrogram(y=y[:480000], sr=16000, n_fft=400, win_length=400,
                                     hop_length=160, n_mels=80, fmin=0, fmax=8000, power=2.0)
db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
out = np.zeros((80, 3001), np.float32)
out[:, :min(db.shape[1], 3001)] = db[:, :3001]
np.savez("name_of_clip_001.npz", out)
```

Use the same parameters at training and inference time. The parity cell in section 10 compares a clip's raw `.npz` statistics against the inference preprocessing so you can catch a mismatch.

### Filenames

If you want the grouped split and the label audit to work, name files `<anything><video id><tags>_NNN.npz`. Only two things matter for the split: segments of the same source video must share everything before `_NNN`, and the numbering suffix must be `_` plus digits. If your names do not carry per-segment tags, the audit reports them as unparseable and training is unaffected. If you have no video grouping at all, the cross-validation splits will leak and the scores will be too optimistic.

### Other changes

| To change | Edit (configuration cell, section 4, unless noted) |
|---|---|
| Clip length | `CLIP_SECONDS`. `TIME_STEPS` follows from it. All `.npz` files must be rebuilt at the new length |
| Sampling rate, hop, mel bins | `SR`, `HOP`, `NUM_MELS`. Rebuild the `.npz` files and keep the inference cell consistent |
| Which labels are excluded | `EXCLUDE_LABELS` |
| Classes and the two heads | `CLASSES`, `HEAD_NAMES`, and in the data cell `CLS_TO_HEADS` and `_LUT`. A new manipulation type needs its own output head, which also means changing `TwoHeadResNet(n_out=2)` and the Y_HEADS construction |
| Training length and folds | `EPOCHS`, `N_FOLDS`, `BATCH_SIZE`, `RUN_CV`, `TRAIN_FINAL` |
| Feature settings | `FEATURE_CFG` (`win` for the splice-step window, `static_norm` as `'bin'` or `'clip'`) |
| Drop rows that disagree with their filename tag | `AUDIT_ACTION = 'drop'` |
| Offline weights instead of ImageNet | `PRETRAINED = False` |
| Raw MP4 dataset for the WavLM probe | `RAW_MP4_DATASET = '<owner>/<dataset>'` |

For the WavLM probe, the MP4 file names must contain the same video IDs as the `.npz` names, and segment `NNN` is assumed to cover seconds `(NNN-1)*30` to `NNN*30`. The cell stops if fewer than 80% of rows match.

## Outputs

| File | Created by |
|---|---|
| `oof_probs.npy` | CV, out-of-fold probabilities |
| `audio_deepfake_resnet18_weights.pt` | final training, weights plus thresholds and configs |
| `wavlm_embeddings.npy`, `wavlm_ok.npy` | optional probe |

Load a saved model with `load_model("audio_deepfake_resnet18_weights.pt")`. It also restores the thresholds and feature settings.

## Known limitations

- The dataset is small. The `cut` class is the hardest and has the fewest examples, so expect high variance between runs and folds.
- Labels are per 30s segment. The position of a cut or dub inside a segment is unknown.
- Zero padding makes clip length visible to the model. Check the shortcut cell in section 5 for class-dependent lengths before trusting results.
- Music-only chunks were excluded from training. There is no voice-activity gate, so they receive arbitrary output.
- The codec test covers only MP3 and AAC at a few bitrates. Other platform re-encoding is untested and can change the signal in ways the training data may not cover.
- Reported numbers should come from the grouped cross-validation report, not from training loss or a random split.

## Ethics and data

The data consists of audio from public social media videos and contains people's voices. Use it for research only, follow the terms of the source platforms and the dataset license, do not redistribute raw media, and do not attempt to identify the people in it. Any result from this project is a statistical estimate and should be treated as one.
