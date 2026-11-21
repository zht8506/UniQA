"""UniQA: one image per row, one randomly selected caption per access."""
import json
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


def resolve_image_path(raw, data_root, path_mappings):
    """Map published placeholder prefixes before resolving ordinary relative paths."""
    for prefix in sorted(path_mappings, key=len, reverse=True):
        if raw == prefix or raw.startswith(prefix.rstrip('/') + '/'):
            return Path(path_mappings[prefix]) / raw[len(prefix):].lstrip('/')
    path = Path(raw).expanduser()
    return path if path.is_absolute() else Path(data_root) / path


def load_records(annotation_files, data_root='.', path_mappings=None):
    # A single filename must not be iterated character by character.
    if isinstance(annotation_files, (str, Path)):
        annotation_files = [annotation_files]
    if not annotation_files:
        raise ValueError('At least one annotation file is required.')
    records, seen = [], set()
    for filename in annotation_files:
        with open(filename, encoding='utf-8') as handle:
            rows = json.load(handle)
        if not isinstance(rows, list):
            raise ValueError(f'{filename}: expected a JSON list.')
        for index, row in enumerate(rows):
            where = f'{filename}, row {index}'
            if not isinstance(row, dict) or not isinstance(row.get('image_path'), str):
                raise ValueError(f'{where}: image_path must be a string.')
            captions = row.get('caption', row.get('prompt'))
            if isinstance(captions, str):
                captions = [captions]
            if not isinstance(captions, list) or not captions or any(
                not isinstance(c, str) or not c.strip() for c in captions
            ):
                raise ValueError(f'{where}: caption (or legacy prompt) must contain nonempty text.')
            path = resolve_image_path(row['image_path'], data_root, path_mappings or {}).resolve()
            if path in seen:
                raise ValueError(f'{where}: duplicate image {path}; merge its captions into one row.')
            seen.add(path)
            records.append((path, captions))
    if not records:
        raise ValueError('The annotation files contain no images.')
    return records


class ImageCaptionDataset(Dataset):
    def __init__(self, records, preprocess, tokenizer):
        self.records, self.preprocess, self.tokenizer = records, preprocess, tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, captions = self.records[index]
        with Image.open(path) as image:
            image = self.preprocess(image.convert('RGB'))
        # Tokenize only the selected caption, preserving uniform sampling per image.
        text = self.tokenizer([random.choice(captions)], truncate=True)[0]
        return image, text
