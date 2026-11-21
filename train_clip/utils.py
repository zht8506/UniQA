import random

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def convert_models_to_fp32(model):
    """Legacy helper: parameters with no gradient are valid."""
    for parameter in model.parameters():
        parameter.data = parameter.data.float()
        if parameter.grad is not None:
            parameter.grad.data = parameter.grad.data.float()


def create_logits(image_features, text_features, logit_scale):
    image_features = F.normalize(image_features.float(), dim=-1)
    text_features = F.normalize(text_features.float(), dim=-1)
    logits = logit_scale.float() * image_features @ text_features.t()
    return logits, logits.t()


def contrastive_loss(image_features, text_features, logit_scale):
    image_logits, text_logits = create_logits(image_features, text_features, logit_scale)
    labels = torch.arange(len(image_features), device=image_features.device)
    return (F.cross_entropy(image_logits, labels) + F.cross_entropy(text_logits, labels)) / 2
