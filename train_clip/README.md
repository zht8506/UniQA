# UniQA pre-training

This directory provides a runnable reconstruction of the pre-training procedure
from the manuscript. It is **not** the archived configuration/log of the original
published run. Use original experiment records to resolve the discrepancies below
before claiming exact numerical reproduction.

## Install and prepare data

Use a dedicated Python environment (Python 3.10–3.12 recommended for OpenAI CLIP):

```bash
python -m pip install -r train_clip/requirements.txt
```

Run commands from the repository root. Download `UniQA_dataset.json` using the
links in the root README. Obtain the corresponding AVA and FLIVE images under
their original dataset licenses. The expected layout is:

```text
datasets/
  UniQA_dataset.json
  images/
    ava/479786.jpg
    flive/...
```

Edit `data.annotation_files`, `data.root` and `data.path_mappings` in
`train_clip/configs/uniqa_vit_b16_paper.yaml` to match your data. Published
annotations use `/dataset_path/ava/...` and `/dataset_path/flive/...`; prefix
mapping replaces these placeholders. Relative image paths are resolved against
`data.root`; other absolute paths are retained. Paths in the config are relative
to the process working directory. This trainer does not infer train/test splits;
only supply the intended pre-training training-set annotations.

Both the released `caption` field and the legacy `prompt` field are accepted:

```json
[
  {"image_path": "/dataset_path/ava/479786.jpg", "caption": ["Description A", "Description B"]},
  {"image_path": "flive/example.jpg", "prompt": "Legacy description"}
]
```

Each row represents one unique image. Each access uniformly samples one caption
from that image; captions are **not** expanded into independent training rows.
Duplicate image rows, empty captions, and missing required fields are errors.
Images are converted to RGB. OpenAI CLIP's preprocessing resizes the shorter side
to 224, center-crops to 224 x 224, and applies CLIP normalization for ViT-B/16.
Captions use CLIP tokenization with truncation enabled, as in the original code.

## Validate before training

```bash
# No model weights are loaded/downloaded; image files need not exist yet.
python -m train_clip.train --config train_clip/configs/uniqa_vit_b16_paper.yaml --dry-run

# Additionally verify that every resolved image file exists.
python -m train_clip.train --config train_clip/configs/uniqa_vit_b16_paper.yaml --dry-run --check-images
```

The report includes unique image count, available caption count, annotation
SHA-256 hashes, batch count and discarded tail size. `--check-images` checks
existence, not image decoding; decoding errors surface during training.

## Train

```bash
# Single process, four visible GPUs. No torchrun is needed.
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m train_clip.train \
  --config train_clip/configs/uniqa_vit_b16_paper.yaml \
  --output experiments/uniqa_pretrain_vit_b16_run1
```

The configured batch 960 is **global**, i.e. 240 images per GPU on four GPUs.
DataParallel encodes images and texts on the selected GPUs, then gathers features
before computing the symmetric cross-entropy over the entire global batch. This
retains cross-GPU negatives. Both encoders and CLIP's logit scale are optimized.
Adam uses a constant learning rate, with no scheduler or gradient accumulation.
AMP is entered inside each replica; FP32 master parameters and GradScaler avoid
manual unsafe gradient conversion. CPU runs require `device: cpu` and
`precision: fp32`. `bf16` is an optional CUDA configuration, not a claim about the
original experiment.

The old entry points `python train_clip/train.py ...` and
`python train_clip/train_multigpu.py ...` call this same trainer. For a one-GPU run,
set `gpu_ids: [0]`. Keeping batch 960 preserves the global contrastive batch but
may exceed memory; reducing it is a different experimental setting. Gradient
accumulation without feature gathering is not an equivalent substitute.

Use `--max-steps 2` only for smoke tests. Such runs are explicitly marked in their
manifest/checkpoint and should not be reported as paper reproduction. Output
directories must be new, so an existing experiment cannot be overwritten.

## Artifacts

- `config.yaml`: resolved configuration.
- `data_manifest.json`: annotation hashes, actual counts, planned steps, environment.
- `metrics.jsonl`: actual step, epoch, loss, learning rate, AMP skipped updates.
- `checkpoint_latest.pt`: latest epoch state, model, Adam, scaler and config.
- `UniQA_weight.pt`: plain CLIP state_dict, compatible with the existing downstream
  loader. This file contains the latest epoch, not a validation-selected model.

Checkpoint files are saved at epoch boundaries (or at a smoke-test limit). Resume
is not implemented by this entry point; the full state is retained for inspection
and future resume support. Failed/incomplete runs remain in their output folder.
Save an environment lock for each full run, e.g. `python -m pip freeze > environment.txt`.

## Configuration provenance and unresolved manuscript discrepancies

| Setting | This configuration | Source/status |
| --- | --- | --- |
| Initialization | OpenAI CLIP ViT-B/16 | Manuscript |
| Image size | 224, shorter-side resize + center crop | Manuscript and CLIP preprocessing |
| Global batch | 960 | Manuscript |
| Epochs | 5 | Manuscript |
| Adam learning rate / weight decay | 5e-6 / 0.2 | Manuscript |
| Adam betas / epsilon | (0.9, 0.98) / 1e-6 | Inherited from released scripts |
| Drop incomplete batch | true | Released multi-GPU script |
| One random caption per image | true | Released Dataset |
| Constant learning rate | true | Released scripts |
| Precision | AMP fp16 with FP32 master weights | New implementation; old code manually cast weights |
| Seed / shuffle / workers | 123 / true / 8 | Explicit new reproducibility defaults |

1. For 273,897 image rows and batch 960, dropping the tail yields **285 steps per
   epoch and 1,425 steps for five epochs**. The revised manuscript corrects the
   previously stated 1,420 steps to 1,425. There are 297 unused tail images per
   epoch (reshuffled each epoch).
2. The local released JSON inspected during this repair contains **273,897 image
   rows and 1,243,838 caption strings**; the manuscript states 1,240,915 captions.
   The trainer computes and records the count of the actual input. Verify which
   annotation version produced the paper's results before editing that claim.
3. The manuscript's phrase “273,897 image-text pairs” should be distinguished from
   the number of available caption strings: the loader samples one caption per
   image each epoch. These are different counts.
4. The earlier scripts used RN50, batch 196/2, and a 50-epoch single-GPU setting.
   Those examples were not the manuscript's ViT-B/16 configuration. This revision
   replaces them with one configurable implementation; it does not establish
   which settings were used historically.

## Validation

```bash
python -m pip install pytest
python -m pytest train_clip/tests -q
```

Tests cover both JSON schemas, prefix mapping, invalid/duplicate metadata,
counts/missing images, cross-chunk contrastive negatives, parameters without
gradients, and actual CPU forward/backward/Adam/checkpoint reload using a tiny
random OpenAI CLIP. No pretrained-weight download is required for those tests.
Four-GPU CUDA execution and full-data convergence must still be verified on the
training host. Caption generation, AIR filtering and downstream experiment
configurations are outside this change.
