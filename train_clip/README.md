# UniQA pre-training

Run the following commands from the repository root.

## 1. Install dependencies

Use Python 3.10–3.12:

```bash
python -m pip install -r train_clip/requirements.txt
```

## 2. Prepare data and configuration

Download `UniQA_dataset.json` using the links in the root README and prepare the
corresponding AVA and FLIVE images:

```text
datasets/
  UniQA_dataset.json
  images/
    ava/479786.jpg
    flive/...
```

Edit `data.annotation_files`, `data.root`, and `data.path_mappings` in
[uniqa_vit_b16_paper.yaml](configs/uniqa_vit_b16_paper.yaml) to match your local
paths. `data.path_mappings` replaces the `/dataset_path/ava` and
`/dataset_path/flive` prefixes in the annotations. Run from the repository root
so that relative paths resolve correctly.

The default configuration uses CLIP ViT-B/16, a global batch size of 960,
5 epochs, and a learning rate of 5e-6. Set `model.name_or_path` to a local OpenAI
CLIP checkpoint if needed.

Optionally check the annotations and image paths before training:

```bash
python -m train_clip.train \
  --config train_clip/configs/uniqa_vit_b16_paper.yaml \
  --dry-run --check-images
```

## 3. Start training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m train_clip.train \
  --config train_clip/configs/uniqa_vit_b16_paper.yaml \
  --output experiments/uniqa_pretrain_vit_b16_run1
```

This launches one process on four GPUs; `torchrun` is not needed. Batch size 960
is the total across all four GPUs (240 per GPU). To use different GPUs, update
`CUDA_VISIBLE_DEVICES` and `runtime.gpu_ids` accordingly.

Use a new output directory for each run. Training saves logs and checkpoints
there, including `UniQA_weight.pt`, the CLIP state dictionary used by the
downstream model, and `checkpoint_latest.pt`, the latest training checkpoint.
