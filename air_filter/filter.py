"""Core AIR: CLIP image-text relevance rank + word-count rank -> Top-K."""
import argparse
import json
import math
from pathlib import Path

import yaml


def select_top_k(captions, similarities, k=4):
    """Both ranks start at zero (best); ties follow original caption order."""
    if k < 1 or len(captions) != len(similarities):
        raise ValueError('k must be positive and scores must match the captions.')
    if any(not math.isfinite(float(score)) for score in similarities):
        raise ValueError('Similarity scores must be finite.')
    if any(not isinstance(text, str) or not text.strip() for text in captions):
        raise ValueError('Each caption must be nonempty text.')
    lengths = [len(text.split()) for text in captions]
    ar_order = sorted(range(len(captions)), key=lambda i: (-similarities[i], i))
    ir_order = sorted(range(len(captions)), key=lambda i: (-lengths[i], i))
    ar = {index: rank for rank, index in enumerate(ar_order)}
    ir = {index: rank for rank, index in enumerate(ir_order)}
    selected = sorted(range(len(captions)), key=lambda i: (ar[i] + ir[i], i))[:k]
    # Match by position, not text: identical comments remain distinct candidates.
    return [captions[index] for index in selected]


def load_clip_scorer(checkpoint, device, text_batch_size=64):
    import clip
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from clip.model import build_model
    from clip.clip import _transform

    if text_batch_size < 1:
        raise ValueError('text_batch_size must be positive.')
    # Supports plain OpenAI CLIP state_dict or train_clip's full checkpoint.
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    model = build_model(state).float()
    # build_model casts some layers to FP16; reload to preserve saved FP32 values.
    model.load_state_dict(state)
    model = model.to(device).eval()
    preprocess = _transform(model.visual.input_resolution)

    def score(image_path, captions):
        if not captions:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in captions):
            raise ValueError(f'Invalid caption for {image_path}')
        with Image.open(image_path) as image:
            image_tensor = preprocess(image.convert('RGB')).unsqueeze(0).to(device)
        with torch.inference_mode():
            image_features = F.normalize(model.encode_image(image_tensor).float(), dim=-1)
            similarities = []
            for start in range(0, len(captions), text_batch_size):
                tokens = clip.tokenize(captions[start:start + text_batch_size], truncate=True).to(device)
                text_features = F.normalize(model.encode_text(tokens).float(), dim=-1)
                similarities.extend((image_features @ text_features.T).squeeze(0).cpu().tolist())
        return similarities

    return score


def filter_rows(rows, image_root, scorer, top_k=4):
    results = []
    for row in rows:
        image_name = row.get('image_path', row.get('image'))
        if not isinstance(image_name, str) or not image_name.strip():
            raise ValueError('Each row must have image_path or image.')
        image_path = Path(image_name)
        if not image_path.is_absolute():
            image_path = Path(image_root) / image_path
        image_path = str(image_path.resolve())
        captions = row['caption']
        if not isinstance(captions, list):
            raise ValueError('caption must be a list of candidate comments.')
        scores = scorer(image_path, captions)
        selected = select_top_k(captions, scores, top_k)
        results.append({'image_path': image_path, 'caption': selected})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    output = Path(config['output_json'])
    if output.exists():
        raise FileExistsError(f'{output} already exists; choose a new output path.')
    if config['top_k'] < 1:
        raise ValueError('top_k must be positive.')
    with open(config['input_json'], encoding='utf-8') as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError('Input JSON must be a list of image/comment records.')
    scorer = load_clip_scorer(config['clip_checkpoint'], config['device'], config['text_batch_size'])
    filtered = filter_rows(rows, config['image_root'], scorer, config['top_k'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(filtered, handle, ensure_ascii=False, indent=2)
    print(f'{len(filtered)} images, {sum(len(row["caption"]) for row in filtered)} selected captions -> {output}')


if __name__ == '__main__':
    main()
