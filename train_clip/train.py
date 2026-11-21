"""Configurable UniQA pre-training; run with python -m train_clip.train."""
import argparse
import hashlib
import json
import math
import os
import platform
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_clip.dataset import ImageCaptionDataset, load_records
from train_clip.utils import contrastive_loss, seed_everything, seed_worker


def load_config(filename):
    with open(filename, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    required = {'model', 'data', 'train', 'runtime'}
    if not isinstance(config, dict) or required - config.keys():
        raise ValueError(f'Config must contain {sorted(required)}.')
    train = config['train']
    for name in ('global_batch_size', 'epochs'):
        if not isinstance(train[name], int) or train[name] < 1:
            raise ValueError(f'{name} must be a positive integer.')
    if train['global_batch_size'] < 2:
        raise ValueError('Contrastive training requires a global batch of at least two.')
    if config['runtime']['precision'] not in ('fp32', 'fp16', 'bf16'):
        raise ValueError('precision must be fp32, fp16 or bf16.')
    return config


def audit_data(config, check_images=False):
    data, train = config['data'], config['train']
    files = data['annotation_files']
    if isinstance(files, str):
        files = [files]
    records = load_records(files, data.get('root', '.'), data.get('path_mappings', {}))
    count = len(records)
    expected = data.get('expected_images')
    if expected is not None and count != expected:
        raise ValueError(f'Expected {expected} images, found {count}. Check the input data/config.')
    batch = train['global_batch_size']
    steps = count // batch if train['drop_last'] else math.ceil(count / batch)
    if steps == 0:
        raise ValueError('No complete batch; reduce global_batch_size or supply more images.')
    if not train['drop_last'] and count % batch == 1:
        raise ValueError('The last batch has one example; enable drop_last for contrastive learning.')
    if check_images:
        missing = [str(path) for path, _ in records if not path.is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} images not found. First paths: {missing[:5]}')
    report = {
        'images': count, 'captions': sum(len(captions) for _, captions in records),
        'sampling': 'one uniformly sampled caption per image per epoch',
        'global_batch_size': batch, 'epochs': train['epochs'],
        'steps_per_epoch': steps, 'total_steps': steps * train['epochs'],
        'dropped_images_per_epoch': count % batch if train['drop_last'] else 0,
        'image_existence_checked': check_images,
        'annotations': [{
            'path': str(Path(path).resolve()),
            'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        } for path in files],
    }
    return records, report


class FeatureEncoder(nn.Module):
    """Gather image/text embeddings across GPUs before forming global negatives."""
    def __init__(self, model, precision):
        super().__init__()
        self.model, self.precision = model, precision

    def forward(self, images, texts):
        # Autocast is thread-local, so enable it inside each DataParallel replica.
        context = nullcontext()
        if images.device.type == 'cuda' and self.precision != 'fp32':
            dtype = torch.float16 if self.precision == 'fp16' else torch.bfloat16
            context = torch.autocast(device_type='cuda', dtype=dtype)
        with context:
            return self.model.encode_image(images), self.model.encode_text(texts)


def train_model(model, preprocess, tokenizer, records, config, output, max_steps=None):
    seed_everything(config['runtime']['seed'])
    runtime, options = config['runtime'], config['train']
    device = torch.device(runtime['device'])
    ids = runtime.get('gpu_ids', [0])
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required by this config. CPU is supported with fp32 for checks.')
        if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= torch.cuda.device_count() for i in ids):
            raise ValueError('gpu_ids must identify distinct visible CUDA devices.')
        device = torch.device(f'cuda:{ids[0]}')
        torch.cuda.set_device(device)
        if runtime['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
            raise ValueError('This CUDA device does not support bf16.')
    elif device.type != 'cpu' or runtime['precision'] != 'fp32':
        raise ValueError('Use device: cpu and precision: fp32 for CPU checks.')
    if device.type == 'cuda' and options['global_batch_size'] % len(ids):
        raise ValueError('global_batch_size must be divisible by the GPU count.')
    model = model.float().to(device)
    model.train()
    encoder = FeatureEncoder(model, runtime['precision'])
    if device.type == 'cuda' and len(ids) > 1:
        encoder = nn.DataParallel(encoder, device_ids=ids, output_device=ids[0])
    generator = torch.Generator().manual_seed(runtime['seed'])
    loader = DataLoader(
        ImageCaptionDataset(records, preprocess, tokenizer),
        batch_size=options['global_batch_size'], shuffle=True,
        num_workers=runtime['num_workers'], drop_last=options['drop_last'],
        pin_memory=device.type == 'cuda', worker_init_fn=seed_worker, generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=options['learning_rate'],
                                 betas=tuple(options['betas']), eps=options['eps'],
                                 weight_decay=options['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and runtime['precision'] == 'fp16')
    output = Path(output)
    step, updates = 0, 0
    for epoch in range(1, options['epochs'] + 1):
        for images, texts in loader:
            images, texts = images.to(device, non_blocking=True), texts.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            image_features, text_features = encoder(images, texts)
            loss = contrastive_loss(image_features, text_features, model.logit_scale.exp())
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at step {step + 1}.')
            scaler.scale(loss).backward()
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            updated = scaler.get_scale() >= old_scale
            updates += int(updated)
            step += 1
            row = {'epoch': epoch, 'step': step, 'optimizer_updates': updates,
                   'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr'],
                   'batch_size': len(images), 'update_skipped': not updated}
            with open(output / 'metrics.jsonl', 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(row) + '\n')
            if step == 1 or step % runtime['log_every'] == 0:
                print(json.dumps(row), flush=True)
            if max_steps and step >= max_steps:
                break
        checkpoint = {'epoch': epoch, 'step': step, 'optimizer_updates': updates,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'scaler_state_dict': scaler.state_dict(), 'config': config,
                      'smoke_test': max_steps is not None}
        # Keep a full checkpoint and a plain state_dict for the existing downstream loader.
        torch.save(checkpoint, output / 'checkpoint_latest.pt')
        torch.save(model.state_dict(), output / 'UniQA_weight.pt')
        if max_steps and step >= max_steps:
            break
    return {'steps': step, 'optimizer_updates': updates}



def load_initial_model(name_or_path, download_root=None):
    import clip
    if not Path(name_or_path).is_file():
        return clip.load(name_or_path, device='cpu', jit=False, download_root=download_root)
    # OpenAI CLIP's file-object fallback may read at EOF after a failed JIT load.
    # Reopen by filename so plain state_dict checkpoints load reliably.
    from clip.model import build_model
    from clip.clip import _transform
    try:
        state = torch.jit.load(str(name_or_path), map_location='cpu').state_dict()
    except RuntimeError:
        state = torch.load(name_or_path, map_location='cpu', weights_only=True)
        if 'model_state_dict' in state:
            state = state['model_state_dict']
    model = build_model(state).float()
    # build_model initially casts some modules to fp16; reload after float() to
    # preserve full precision when initializing from our FP32 saved weights.
    model.load_state_dict(state)
    return model, _transform(model.visual.input_resolution)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='train_clip/configs/uniqa_vit_b16_paper.yaml')
    parser.add_argument('--output', help='New output directory; existing directories are not overwritten.')
    parser.add_argument('--dry-run', action='store_true', help='Validate metadata and report steps; do not load CLIP.')
    parser.add_argument('--check-images', action='store_true', help='Also check all image paths during a dry run.')
    parser.add_argument('--max-steps', type=int, help='Stop early for a smoke test; not a paper reproduction.')
    args = parser.parse_args(argv)
    if args.max_steps is not None and args.max_steps < 1:
        parser.error('--max-steps must be positive.')
    config = load_config(args.config)
    records, report = audit_data(config, args.check_images or not args.dry_run)
    print(json.dumps(report, indent=2), flush=True)
    if args.dry_run:
        return
    import clip
    seed_everything(config['runtime']['seed'])
    model, preprocess = load_initial_model(config['model']['name_or_path'],
                                           config['model'].get('download_root'))
    if model.visual.input_resolution != config['model']['image_size']:
        raise ValueError('Loaded model resolution differs from configured image_size.')
    output = Path(args.output or config['runtime']['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    report['environment'] = {'python': platform.python_version(), 'torch': torch.__version__,
                             'numpy': np.__version__, 'cuda': torch.version.cuda,
                             'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')}
    report['smoke_test_max_steps'] = args.max_steps
    (output / 'data_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    train_model(model, preprocess, clip.tokenize, records, config, output, args.max_steps)


if __name__ == '__main__':
    main()
