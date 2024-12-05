r"""CNNIQA Model.

Zheng, Heliang, Huan Yang, Jianlong Fu, Zheng-Jun Zha, and Jiebo Luo.
"Learning conditional knowledge distillation for degraded-reference image
quality assessment." In Proceedings of the IEEE/CVF International Conference
on Computer Vision (ICCV), pp. 10242-10251. 2021.

Ref url: https://github.com/lidq92/CNNIQA
Re-implemented by: Chaofeng Chen (https://github.com/chaofengc) with modification:
    - We use 3 channel RGB input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import load_pretrained_network, get_url_from_name

default_model_urls = {
    'koniq10k': get_url_from_name('CNNIQA_koniq10k-e6f14c91.pth')
}


@ARCH_REGISTRY.register()
class CNNIQA(nn.Module):
    r"""CNNIQA model.

    Args:
        ker_size (int): Kernel size.
        n_kers (int): Number of kernels.
        n1_nodes (int): Number of n1 nodes.
        n2_nodes (int): Number of n2 nodes.
        pretrained (str): Pretrained model name.
        pretrained_model_path (str): Pretrained model path.
    """

    def __init__(
        self,
        ker_size=7,
        n_kers=50,
        n1_nodes=800,
        n2_nodes=800,
        pretrained='koniq10k',
        pretrained_model_path=None,
    ):
        super(CNNIQA, self).__init__()

        self.conv1 = nn.Conv2d(3, n_kers, ker_size)
        self.fc1 = nn.Linear(2 * n_kers, n1_nodes)
        self.fc2 = nn.Linear(n1_nodes, n2_nodes)
        self.fc3 = nn.Linear(n2_nodes, 1)
        self.dropout = nn.Dropout()

        if pretrained_model_path is None and pretrained is not None:
            pretrained_model_path = default_model_urls[pretrained]

        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, True, 'params')

    def forward(self, x):
        r"""Compute IQA using CNNIQA model.

        Args:
            x (torch.Tensor): An input tensor with (N, C, H, W) shape. RGB channel order for colour images.

        Returns:
            torch.Tensor: Value of CNNIQA model.
        """
        h = self.conv1(x)

        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))
        h = torch.cat((h1, h2), 1)  # max-min pooling
        h = h.squeeze(3).squeeze(2)

        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))

        q = self.fc3(h)
        return q
