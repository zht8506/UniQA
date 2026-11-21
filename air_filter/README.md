# AIR caption filtering

A minimal example of the length-based AIR strategy. Load a **fine-tuned aesthetics-aware
CLIP** checkpoint, compute cosine similarity between each image and its raw candidate
comments, and select Top-4 using the sum of relevance and length ranks.

## Input and configuration

Use the CLIP environment described in `train_clip/requirements.txt`:

```bash
python -m pip install -r train_clip/requirements.txt
```

Edit the example paths in `config.yaml`. `clip_checkpoint` should be the fine-tuned
`CLIP_aes` weight, not vanilla CLIP. OpenAI CLIP plain state_dict and the full
`model_state_dict` checkpoint saved by our pre-training script are supported;
architecture/resolution are inferred from the weights. OpenCLIP/Hugging Face
checkpoints with different parameter names and TorchScript archives are not supported.

Input is a JSON list containing the raw **training-set** AVA comments:

```json
[
  {"image": "479786.jpg", "caption": ["Nice colors.", "The balanced composition draws attention to the subject."]}
]
```

`image_path` is also accepted. Relative paths resolve against `image_root`;
absolute paths are used directly. Replace any placeholder prefixes in the input
if needed. Both input paths and config paths are interpreted from the current
working directory. Only the supplied rows are processed; this script does not
create dataset splits.

## Run from the repository root

```bash
CUDA_VISIBLE_DEVICES=0 python air_filter/filter.py
# Or use a custom configuration:
python air_filter/filter.py --config air_filter/config.yaml
```

For a CPU run, set `device: cpu`. The model is loaded once and used in evaluation
mode without gradients. Each image is encoded once; comments are encoded in
batches of 64 (configurable). CLIP's original preprocessing and tokenizer are used,
with text truncation to CLIP's context length enabled.

## Ranking rule

For each image independently:

1. Normalize image/text features; compute their cosine similarities.
2. Sort similarity descending to obtain **AR**, starting at rank 0.
3. Count whitespace-separated words in each full comment; sort descending to obtain **IR**.
4. Compute **AIR = AR + IR** (alpha = beta = 1); sort ascending and retain Top-4.

There is no need to multiply by CLIP's learned positive logit scale: it does not
change the within-image ordering. Relevance is computed from the image and the
candidate human comment, not text-text similarity with a generated description.

Ties use the original candidate order at every sorting stage. Duplicate strings
are treated as distinct positions; no string-based rank lookup is used. If fewer
than four comments exist, all are returned in AIR order. An empty list remains
empty; empty strings are rejected. Word counting uses `split()` rather than the
old `split(" ")`, so repeated whitespace is not counted as extra words. Explicit
stable tie-breaking may differ from the old NumPy default sort for tied scores.

Output uses the pre-training-compatible format:

```json
[{"image_path": "/path/to/ava/images/479786.jpg", "caption": ["Selected comment."]}]
```

The output contains only the selected human comments. It does not add MOS prefixes
or merge LLaVA-generated descriptions. Existing output files are not overwritten.
This is the core example only: no sharding, resume or failure-log machinery.
