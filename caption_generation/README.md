# FLIVE / AVA caption generation

Generate captions for FLIVE and AVA with LLaVA-1.5-7B. By default, each FLIVE
image receives three captions and each AVA image receives one caption.

## 1. Install dependencies

Install LLaVA and PyYAML in a separate LLaVA environment:

```bash
python -m pip install -e /path/to/LLaVA
python -m pip install PyYAML
```

## 2. Configure paths

Edit [config.yaml](config.yaml):

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

Prompt texts and generation parameters are in `config.yaml`. Select prompts
with each dataset's `prompt_ids`. Defaults are `flive_expert`,
`flive_descriptors`, and `flive_observer` for FLIVE, and `ava_expert` for AVA.
The default generation settings are temperature 0, top_p 1.0, num_beams 1,
max_new_tokens 1024, and conversation mode `llava_v1`.

## 3. Run

Run from the UniQA repository root on a CUDA GPU:

```bash
# Generate captions for both datasets.
CUDA_VISIBLE_DEVICES=0 python caption_generation/generate.py --dataset both

# Or run one dataset.
CUDA_VISIBLE_DEVICES=0 python caption_generation/generate.py --dataset flive
CUDA_VISIBLE_DEVICES=0 python caption_generation/generate.py --dataset ava
```

The default output files are `outputs/flive_captions.json` and
`outputs/ava_captions.json`, using this format:

```json
[{"image_path": "/path/to/image.jpg", "caption": ["Generated caption."]}]
```
