# Data preparation

Run Caption generation and AIR filtering with the scripts below. All commands
are run from the UniQA repository root.

## Caption generation

Generate captions for FLIVE and AVA with LLaVA-1.5-7B. By default, each FLIVE
image receives three captions and each AVA image receives one caption.

### 1. Install dependencies

Install LLaVA and PyYAML in a separate LLaVA environment:

```bash
python -m pip install -e /path/to/LLaVA
python -m pip install PyYAML
```

### 2. Configure paths

Edit [configs/caption_generation.yaml](configs/caption_generation.yaml):

- `model_path`: local LLaVA-1.5-7B checkpoint directory.
- `datasets.<name>.csv_path`: image metadata CSV.
- `datasets.<name>.image_root`: image directory.
- `datasets.<name>.output`: output JSON path; use a new path for each run.

Expected CSV columns:

| Dataset | Image column | MOS column | MOS range |
| --- | --- | --- | --- |
| FLIVE | `name` | `mos` | 0–100 |
| AVA | `img_name` | `MOS` | 0–10 |

Use the AVA pre-training train split. Image and MOS column names can be changed
in the configuration.

Prompt texts and generation parameters are in `configs/caption_generation.yaml`. Select prompts
with each dataset's `prompt_ids`. Defaults are `flive_expert`,
`flive_descriptors`, and `flive_observer` for FLIVE, and `ava_expert` for AVA.
The default generation settings are temperature 0, top_p 1.0, num_beams 1,
max_new_tokens 1024, and conversation mode `llava_v1`.

### 3. Run

Run from the UniQA repository root on a CUDA GPU:

```bash
# Generate captions for both datasets.
CUDA_VISIBLE_DEVICES=0 python data_preparation/generate_captions.py --dataset both

# Or run one dataset.
CUDA_VISIBLE_DEVICES=0 python data_preparation/generate_captions.py --dataset flive
CUDA_VISIBLE_DEVICES=0 python data_preparation/generate_captions.py --dataset ava
```

The default output files are `outputs/flive_captions.json` and
`outputs/ava_captions.json`, using this format:

```json
[{"image_path": "/path/to/image.jpg", "caption": ["Generated caption."]}]
```

## AIR filtering

Load a fine-tuned `CLIP_aes` checkpoint and select Top-4 AVA comments per image
using the sum of image-text relevance and word-count ranks.

### 1. Install dependencies

Use the UniQA CLIP training environment:

```bash
python -m pip install -r train_clip/requirements.txt
```

### 2. Configure paths and input

Edit [configs/air_filter.yaml](configs/air_filter.yaml):

- `clip_checkpoint`: fine-tuned `CLIP_aes` weight (OpenAI CLIP state dictionary or
  a training checkpoint containing `model_state_dict`).
- `input_json`: AVA training-set candidate comments.
- `image_root`: base directory for relative image paths.
- `output_json`: output path; use a new path for each run.
- `top_k`: number of comments to retain, default 4.
- `device`: `cuda:0` or `cpu`.

Input JSON format:

```json
[
  {"image": "479786.jpg", "caption": ["Nice colors.", "The balanced composition draws attention to the subject."]}
]
```

`image_path` can also be used instead of `image`. Absolute image paths are used
directly; relative paths resolve against `image_root`.

### 3. Run

Run from the UniQA repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python data_preparation/filter_captions.py

# Or specify a configuration file.
CUDA_VISIBLE_DEVICES=0 python data_preparation/filter_captions.py \
  --config data_preparation/configs/air_filter.yaml
```

The default output is `outputs/ava_captions_air_top4.json`:

```json
[{"image_path": "/path/to/ava/images/479786.jpg", "caption": ["Selected comment."]}]
```
