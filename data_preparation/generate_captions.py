"""Minimal shared FLIVE/AVA caption generation with LLaVA-1.5."""
import argparse
from bisect import bisect_right
import csv
import json
import math
from pathlib import Path

import yaml


def mos_level(score, boundaries, levels):
    """Left-closed bins; the last bin also includes the maximum MOS."""
    if len(boundaries) != len(levels) + 1 or any(
        a >= b for a, b in zip(boundaries, boundaries[1:])
    ):
        raise ValueError('Provide increasing MOS boundaries and one label per bin.')
    if not math.isfinite(score) or not boundaries[0] <= score <= boundaries[-1]:
        raise ValueError(f'MOS {score} is outside [{boundaries[0]}, {boundaries[-1]}].')
    return levels[min(bisect_right(boundaries, score) - 1, len(levels) - 1)]


def read_samples(dataset):
    with open(dataset['csv_path'], newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    samples, seen = [], set()
    for row in rows:
        image = Path(row[dataset['image_column']])
        if not image.is_absolute():
            image = Path(dataset['image_root']) / image
        image = image.resolve()
        if image in seen:
            raise ValueError(f'Duplicate image in metadata: {image}')
        seen.add(image)
        level = mos_level(float(row[dataset['mos_column']]),
                          dataset['boundaries'], dataset['levels'])
        samples.append((str(image), level))
    if not samples:
        raise ValueError(f"No images in {dataset['csv_path']}")
    return samples


def load_captioner(config):
    # LLaVA imports are lazy so metadata/prompt checks do not need GPU packages.
    import torch
    from PIL import Image
    from transformers import set_seed
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token, KeywordsStoppingCriteria
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle

    set_seed(config['seed'])
    tokenizer, model, image_processor, _ = load_pretrained_model(
        config['model_path'], config.get('model_base'),
        get_model_name_from_path(config['model_path']),
        device='cuda', device_map={'': 'cuda:0'},
    )
    model.eval()

    def caption(image_path, question):
        image_token = DEFAULT_IMAGE_TOKEN
        if model.config.mm_use_im_start_end:
            image_token = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN
        conversation = conv_templates[config['conv_mode']].copy()
        conversation.append_message(conversation.roles[0], image_token + '\n' + question)
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(model.device)
        with Image.open(image_path) as image:
            image_tensor = process_images([image.convert('RGB')], image_processor, model.config)
        # LLaVA can return either a tensor or a list when aspect-ratio processing differs.
        if isinstance(image_tensor, list):
            image_tensor = [value.to(model.device, dtype=torch.float16) for value in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        stop = conversation.sep if conversation.sep_style != SeparatorStyle.TWO else conversation.sep2
        stopping = KeywordsStoppingCriteria([stop], tokenizer, input_ids)
        generation = dict(config['generation'])
        generation['do_sample'] = generation['temperature'] > 0
        if not generation['do_sample']:
            # Sampling-only settings are inactive for greedy decoding.
            generation.pop('temperature')
            generation.pop('top_p')
        with torch.inference_mode():
            output_ids = model.generate(input_ids, images=image_tensor,
                                        stopping_criteria=[stopping], **generation)
        # Original LLaVA returns the input prefix; some later versions return only new tokens.
        length = input_ids.shape[1]
        if output_ids.shape[1] >= length and torch.equal(output_ids[:, :length], input_ids):
            output_ids = output_ids[:, length:]
        result = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if stop and result.endswith(stop):
            result = result[:-len(stop)].strip()
        if not result:
            raise ValueError(f'Empty caption for {image_path}')
        return result

    return caption


def generate_samples(samples, templates, captioner):
    return [
        {'image_path': image, 'caption': [captioner(image, text.format(level=level)) for text in templates]}
        for image, level in samples
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).resolve().parent / 'configs' / 'caption_generation.yaml'))
    parser.add_argument('--dataset', choices=['flive', 'ava', 'both'], default='both')
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    names = ['flive', 'ava'] if args.dataset == 'both' else [args.dataset]
    # Validate inputs before loading the model. The model is loaded once for both datasets.
    prepared = []
    for name in names:
        dataset = config['datasets'][name]
        samples = read_samples(dataset)
        templates = [config['prompts'][key] for key in dataset['prompt_ids']]
        if not templates:
            raise ValueError(f'{name}: at least one prompt is required.')
        output = Path(dataset['output'])
        if output.exists():
            raise FileExistsError(f'{output} already exists; choose a new output path.')
        if any(output.resolve() == item[3].resolve() for item in prepared):
            raise ValueError('Datasets must use different output paths.')
        prepared.append((name, samples, templates, output))
    captioner = load_captioner(config)
    for name, samples, templates, output in prepared:
        results = generate_samples(samples, templates, captioner)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        print(f'{name}: {len(results)} images, {len(results) * len(templates)} captions -> {output}')


if __name__ == '__main__':
    main()
