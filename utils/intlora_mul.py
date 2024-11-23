import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.quant_utils import batch_mse,batch_max,lp_loss,round_ste
from utils.quant_layer import UniformAffineQuantizer



class IntLoRA_MUL(nn.Module):
    # The implementation of our IntLoRA-MUL
    def __init__(self, org_module, n_bits=8, lora_bits=8, symmetric=False,channel_wise=True, rank=4, activation_params=None):
        super(IntLoRA_MUL, self).__init__()

        self.n_bits = n_bits
        self.lora_bits = lora_bits
        self.sym = symmetric
        self.scale_method = 'mse'
        self.always_zero = False
        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1
        self.channel_wise = channel_wise
        self.inited=False
        self.lora_levels = self.n_levels

        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features


        # save original weights and bias and keep them intact
        self.ori_weight_shape = org_module.weight.shape

        self.ori_weight = org_module.weight.view(self.out_features,-1).data.clone() # reshape here
        self.ori_bias = None if org_module.bias is None else org_module.bias.data.clone()

        self.act_quant_params = activation_params
        if self.act_quant_params is not None:
            self.act_quantizer = UniformAffineQuantizer(**self.act_quant_params)


        # quant lora quant here ===========================
        self.quant_lora_weights = True
        self.double_inited = False
        self.double_quant_delta = torch.nn.Parameter(torch.zeros(self.out_features,1))
        self.double_quant_zero_point = torch.nn.Parameter(torch.zeros(self.out_features,1))

        r = rank
        self.alpha = 1.5
        self.loraA = nn.Linear(org_module.in_features, r, bias=False)
        self.loraB = nn.Linear(r, org_module.out_features, bias=False)
        nn.init.kaiming_uniform_(self.loraA.weight, a=math.sqrt(5))
        nn.init.zeros_(self.loraB.weight)

        # init the auxiliary matrix R in IntLoRA
        m = torch.distributions.laplace.Laplace(loc=torch.tensor([0.]),scale=torch.tensor([0.5]))
        aux_R = m.sample((org_module.out_features,org_module.in_features))[:,:,0]
        self.register_buffer('aux_R', aux_R)
        self.aux_R = self.aux_R.to(self.ori_weight.device).detach()


    def forward(self, input: torch.Tensor):
        if self.inited is False:
            aux_R_abs_max  = torch.minimum(self.aux_R.max(dim=-1,keepdim=True)[0].abs(),self.aux_R.min(dim=-1,keepdim=True)[0].abs()).detach()
            ori_weight_abs_max = torch.maximum(self.ori_weight.max(dim=-1,keepdim=True)[0].abs(),self.ori_weight.min(dim=-1,keepdim=True)[0].abs()).detach()
            self.aux_R = ((ori_weight_abs_max)**self.alpha/(aux_R_abs_max+1e-8)**self.alpha)*self.aux_R

            ori_weight = self.ori_weight - self.aux_R
            delta, zero_point = self.init_quantization_scale(ori_weight, self.channel_wise,self.n_bits,self.sym)
            self.register_buffer('weight_quant_delta', delta)
            self.register_buffer('weight_quant_zero_point', zero_point)
            ori_weight_round = round_ste(ori_weight / self.weight_quant_delta) + self.weight_quant_zero_point
            if self.sym:
                ori_weight_round = torch.clamp(ori_weight_round, -self.n_levels - 1, self.n_levels)
            else:
                ori_weight_round = torch.clamp(ori_weight_round, 0, self.n_levels - 1)

            # delete the FP weights and save the int weights
            self.register_buffer('ori_weight_round', ori_weight_round)  # int weight and keep it intact
            self.ori_weight = None
            torch.cuda.empty_cache()
            self.inited = True

        ori_weight_int = self.ori_weight_round - self.weight_quant_zero_point 

        lora_weight = (self.aux_R + (self.loraB.weight @ self.loraA.weight)) / \
                          torch.where(ori_weight_int == 0, torch.tensor(1).to(ori_weight_int.device), ori_weight_int)
        weight_updates = self.weight_quant_delta + lora_weight # broad-cast


        if self.quant_lora_weights:
            if self.double_inited is False:
                delta, zero_point = self.init_quantization_scale(weight_updates, True, self.lora_bits)
                with torch.no_grad():
                    self.double_quant_delta.copy_(delta)
                    self.double_quant_zero_point.copy_(zero_point)
                self.double_inited = True

            weight_updates_round = round_ste(weight_updates / self.double_quant_delta) + self.double_quant_zero_point
            if self.sym:
                weight_updates_round = torch.clamp(weight_updates_round, -self.lora_levels - 1, self.lora_levels)
            else:
                weight_updates_round = torch.clamp(weight_updates_round, 0, self.lora_levels - 1)

            weight_updates_int = (weight_updates_round - self.double_quant_zero_point)
            weight_int_mul = weight_updates_int * ori_weight_int # simulated INT multiply
            weight = self.double_quant_delta * weight_int_mul

        else:
            weight = weight_updates*ori_weight_int

        bias = self.ori_bias

        # do activation quantization
        if self.act_quant_params is not None:
           input = self.act_quantizer(input)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        return out



    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False, n_bits: int = 8, sym: bool= False):
        n_levels = 2 ** n_bits if not sym else 2 ** (n_bits - 1) - 1
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            if 'max' in self.scale_method:
                delta, zero_point = batch_max(x_clone.view(n_channels, -1), sym, 2 ** n_bits,
                                              self.always_zero)

            elif 'mse' in self.scale_method:
                delta, zero_point = batch_mse(x_clone.view(n_channels, -1), sym, 2 ** n_bits,
                                              self.always_zero)

            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 3:
                delta = delta.view(-1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if 'scale' in self.scale_method:
                    x_min = x_min * (n_bits + 2) / 8
                    x_max = x_max * (n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if sym:
                    # x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax
                    delta = x_absmax / n_levels
                else:
                    delta = float(x.max().item() - x.min().item()) / (n_levels - 1)
                if delta < 1e-8:
                    delta = 1e-8

                zero_point = round(-x_min / delta) if not (sym or self.always_zero) else 0
                delta = torch.tensor(delta).type_as(x)

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                for i in range(80):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min,n_bits,sym)
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** n_bits - 1) \
                            if not self.always_zero else new_max / (2 ** n_bits - 1)
                        zero_point = (- new_min / delta).round() if not self.always_zero else 0
            else:
                raise NotImplementedError

        return delta, zero_point

    def quantize(self, x, max, min,n_bits,sym):
        n_levels = 2 ** n_bits if not sym else 2 ** (n_bits - 1) - 1
        delta = (max - min) / (2 ** n_bits - 1) if not self.always_zero else max / (2 ** n_bits - 1)
        zero_point = (- min / delta).round() if not self.always_zero else 0
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0,n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q
