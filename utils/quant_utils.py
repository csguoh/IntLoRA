import torch
import torch.nn.functional as F
from torch import nn
from diffusers.utils import deprecate, logging
from diffusers.utils.import_utils import is_xformers_available
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


def batch_mse(x: torch.Tensor,
              symmetric: bool = False,
              level: int = 256,
              always_zero: bool = False
              ) -> [torch.Tensor, torch.Tensor]:
    x_min, x_max = torch.min(x, dim=-1, keepdim=True)[0], torch.max(x, dim=-1, keepdim=True)[0]  # [d_out]
    delta, zero_point = torch.zeros_like(x_min), torch.zeros_like(x_min)
    s = torch.full((x.shape[0], 1), 30000, dtype=x.dtype, device=x.device)
    for i in range(80):
        new_min = x_min * (1. - (i * 0.01))
        new_max = x_max * (1. - (i * 0.01))
        new_delta = (new_max - new_min) / (level - 1)
        new_zero_point = torch.round(-new_min / new_delta) if not (symmetric or always_zero) else torch.zeros_like(new_delta)
        NB, PB = -level // 2 if symmetric and not always_zero else 0, \
            level // 2 - 1 if symmetric and not always_zero else level - 1
        x_q = torch.clamp(torch.round(x / new_delta) + new_zero_point, NB, PB)
        x_dq = new_delta * (x_q - new_zero_point)
        new_s = (x_dq - x).abs().pow(2.4).mean(dim=-1, keepdim=True)

        update_mask = new_s < s
        delta[update_mask] = new_delta[update_mask]
        zero_point[update_mask] = new_zero_point[update_mask]
        s[update_mask] = new_s[update_mask]

    return delta.squeeze(), zero_point.squeeze()


def batch_max(x: torch.Tensor,
              symmetric: bool = False,
              level: int = 256,
              always_zero: bool = False
              ) -> [torch.Tensor, torch.Tensor]:
    x_min, x_max = torch.min(x, dim=-1, keepdim=True)[0], torch.max(x, dim=-1, keepdim=True)[0]  # [d_out]

    x_absmax = torch.max(torch.cat([x_min.abs(), x_max], dim=-1), dim=-1, keepdim=True)[0]

    if symmetric:
        # x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax
        delta = x_absmax / level
    else:
        delta = (x_max - x_min) / (level - 1)

    delta = torch.clamp(delta, min=1e-8)
    zero_point = torch.round(-x_min / delta) if not (symmetric or always_zero) else 0

    return delta.squeeze(), zero_point.squeeze()



def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x



def lp_loss(pred, tgt, p=2.0, reduction='none'):
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()



