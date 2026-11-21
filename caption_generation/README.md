# FLIVE / AVA caption generation

A minimal shared example extracted from the supplied UniQA LLaVA scripts.
It loads LLaVA once and generates three captions per FLIVE image and one caption
per AVA image by default. It does not implement sharding, resume, or failure logs;
an error stops the run. Outputs are written when each dataset finishes.

## Setup

Use the supplied LLaVA-1.5 source in a **separate environment** from the UniQA
pre-training environment. Its dependency versions differ from the training stack.
Install that LLaVA source and PyYAML, for example:

```bash
# In the LLaVA environment. Substitute the location of the supplied LLaVA source.
python -m pip install -e /path/to/LLaVA
python -m pip install PyYAML
```

Edit `config.yaml`: replace the example LLaVA checkpoint, image folders and CSV
paths. The example uses `/path/to/llava-v1.5-7b`; 13B can be selected by changing
this path. Run on a CUDA GPU with sufficient memory. Image preprocessing uses
LLaVA's image processor and checkpoint configuration, without an added resize step.

CSV columns:

- FLIVE: `name,mos`, with MOS in [0, 100].
- AVA: `img_name,MOS`, with MOS in [0, 10]. Supply the **pre-training train split
  only**, not the complete AVA metadata. This script does not create dataset splits.

## Run from the UniQA repository root

```bash
CUDA_VISIBLE_DEVICES=0 python caption_generation/generate.py --dataset both
# Or select one dataset:
CUDA_VISIBLE_DEVICES=0 python caption_generation/generate.py --dataset flive
CUDA_VISIBLE_DEVICES=0 python caption_generation/generate.py --dataset ava
```

The script writes `outputs/flive_captions.json` and `outputs/ava_captions.json`:

```json
[{"image_path": "/path/to/image.jpg", "caption": ["Generated caption."]}]
```

Each image has a list of captions, matching the pre-training loader. These are
only the generated descriptions; no AIR-selected human comments are added here.
Existing output files are not overwritten; choose a new path for a new run.

## Prompts and generation settings

All distinct FLIVE/AVA prompt variants from the supplied scripts are listed in
`config.yaml`. Change a dataset's `prompt_ids` to select variants:

| Dataset | Default | Other supplied variants |
| --- | --- | --- |
| FLIVE | `flive_expert`, `flive_descriptors`, `flive_observer` | — |
| AVA | `ava_expert` | `ava_observer`, `ava_unguided`, `ava_rate_and_explain` |

`ava_unguided` is the commented alternative in the original direct-generation
script; `ava_rate_and_explain` is its active prompt. Neither receives MOS guidance.
The original AVA 1–4 scripts split images across jobs using the same expert prompt;
they do not represent four distinct prompts.

Generation follows the supplied scripts: temperature 0 (greedy decoding), top_p
1.0, num_beams 1, max_new_tokens 1024, use_cache true, conversation `llava_v1`.
For greedy generation, sampling-only temperature/top_p arguments are not passed
to `generate`. A fixed seed of 123 is a new explicit default; the original scripts
did not set a seed.

FLIVE labels are `bad, poor, fair, good, perfect`. AVA preserves the supplied
scripts' `very bad, poor, medium, good, perfect`; this differs from the manuscript's
unified wording. Labels are configurable. Bins include the lower bound, and the
final bin also includes 100 (FLIVE) or 10 (AVA). Out-of-range/NaN MOS is rejected.
Missing whitespace before `quality` in FLIVE prompts has been fixed. The historical
AVA alternative wording is retained, including the label order in its unguided
rating prompt. This example does not establish which historical variant produced
the released captions; the supplied files mix 7B/13B and guided/unguided experiments.
