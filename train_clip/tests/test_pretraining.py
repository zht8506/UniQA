import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from train_clip.dataset import ImageCaptionDataset, load_records
from train_clip.train import audit_data, load_config, load_initial_model, train_model
from train_clip.utils import contrastive_loss, convert_models_to_fp32


CONFIG = Path(__file__).resolve().parents[1] / 'configs/uniqa_vit_b16_paper.yaml'


def write_rows(tmp_path, rows):
    path = tmp_path / 'annotations.json'
    path.write_text(json.dumps(rows))
    return path


def test_published_and_legacy_schema_and_path_mapping(tmp_path):
    path = write_rows(tmp_path, [
        {'image_path': '/dataset_path/ava/a.jpg', 'caption': ['one', 'two']},
        {'image_path': 'b.jpg', 'prompt': 'legacy'},
    ])
    records = load_records(str(path), tmp_path, {'/dataset_path/ava': str(tmp_path / 'ava')})
    assert records == [(tmp_path / 'ava/a.jpg', ['one', 'two']), (tmp_path / 'b.jpg', ['legacy'])]


@pytest.mark.parametrize('rows', [
    [{'image_path': 'a.jpg', 'caption': []}],
    [{'image_path': 'a.jpg', 'caption': ['']}],
    [{'image_path': 'a.jpg', 'caption': 'a'}, {'image_path': './a.jpg', 'caption': 'b'}],
])
def test_bad_metadata_fails_before_training(tmp_path, rows):
    with pytest.raises(ValueError):
        load_records(write_rows(tmp_path, rows), tmp_path)


def test_counts_and_missing_image_validation(tmp_path):
    path = write_rows(tmp_path, [{'image_path': f'{i}.jpg', 'caption': ['a', 'b']} for i in range(6)])
    config = load_config(CONFIG)
    config['data'].update(annotation_files=[str(path)], root=str(tmp_path), expected_images=6)
    config['train'].update(global_batch_size=4, epochs=5)
    _, report = audit_data(config)
    assert (report['images'], report['captions'], report['total_steps']) == (6, 12, 5)
    assert report['dropped_images_per_epoch'] == 2
    with pytest.raises(FileNotFoundError):
        audit_data(config, check_images=True)


def test_global_contrastive_loss_uses_negatives_across_chunks():
    torch.manual_seed(123)
    image = torch.randn(8, 16, requires_grad=True)
    text = torch.randn(8, 16, requires_grad=True)
    scale = torch.tensor(2., requires_grad=True)
    full = contrastive_loss(image, text, scale)
    gathered = contrastive_loss(torch.cat(image.chunk(4)), torch.cat(text.chunk(4)), scale)
    local_only = sum(contrastive_loss(i, t, scale) for i, t in zip(image.chunk(4), text.chunk(4))) / 4
    assert torch.allclose(full, gathered)
    assert not torch.allclose(full, local_only)
    full.backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (image, text, scale))


def test_fp32_conversion_handles_absent_gradient():
    model = torch.nn.Linear(2, 2).half()
    convert_models_to_fp32(model)
    assert model.weight.dtype == torch.float32


def test_real_clip_cpu_training_and_checkpoint(tmp_path):
    # Exercise actual OpenAI CLIP, transforms, tokenization, Adam and checkpoint reload;
    # use a tiny random model to avoid downloading production weights or training on user data.
    import clip
    from clip.model import CLIP
    from clip.clip import _transform
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = CLIP(embed_dim=32, image_resolution=16, vision_layers=1,
                 vision_width=64, vision_patch_size=8, context_length=77,
                 vocab_size=49408, transformer_width=64, transformer_heads=1,
                 transformer_layers=1)
    records = []
    for i in range(4):
        path = tmp_path / f'{i}.png'
        Image.fromarray(np.full((20, 24), i * 60, dtype=np.uint8)).save(path)
        records.append((path, [f'quality level {i}', f'image number {i}']))
    config = load_config(CONFIG)
    config['runtime'].update(device='cpu', precision='fp32', num_workers=0, log_every=1)
    config['train'].update(global_batch_size=4, epochs=2)
    before = model.visual.conv1.weight.detach().clone()
    result = train_model(model, _transform(16), clip.tokenize, records, config, tmp_path)
    assert result == {'steps': 2, 'optimizer_updates': 2}
    assert not torch.equal(before, model.visual.conv1.weight)
    checkpoint = torch.load(tmp_path / 'checkpoint_latest.pt', weights_only=True)
    assert checkpoint['epoch'] == 2 and not checkpoint['smoke_test']
    assert checkpoint['optimizer_state_dict']['state']
    restored, _ = load_initial_model(str(tmp_path / 'UniQA_weight.pt'))
    assert torch.equal(restored.visual.conv1.weight, model.visual.conv1.weight)
    rows = [json.loads(line) for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
    assert len(rows) == 2 and all(np.isfinite(row['loss']) for row in rows)
