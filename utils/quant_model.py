import torch
import torch.nn.functional as F
from torch import nn
from utils.intlora_mul import IntLoRA_MUL
from utils.intlora_shift import IntLoRA_SHIFT
from utils.quant_layer import QuantLayerNormal



class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input



def check_in_special(ss, special_list):
    for itm in special_list:
        if itm in ss:
            return True
    return False



class QuantUnetWarp(nn.Module):
    def __init__(self, model: nn.Module, args):
        super().__init__()
        self.model = model
        self.intlora_type = args.intlora
        lora_quant_params = {'n_bits': args.nbits, 'lora_bits':args.nbits, 'symmetric':False, 'channel_wise':True, 'rank':args.rank}
        other_quant_params = {'n_bits': args.nbits, 'symmetric': False, 'channel_wise': True, 'scale_method': 'mse'}
        activation_quant_params = {'n_bits': args.act_nbits, 'symmetric': False, 'channel_wise': False, 'scale_method': 'mse','leaf_param': True} if args.use_activation_quant else None
        special_list = ['to_q','to_k','to_v','to_out']
        assert self.intlora_type in ['MUL','SHIFT']
        self.IntLoRALayer = IntLoRA_MUL if self.intlora_type == 'MUL' else IntLoRA_SHIFT

        self.quant_module_refactor(self.model, lora_quant_params, other_quant_params, activation_quant_params, special_list)


    def quant_module_refactor(self, module: nn.Module, lora_quant_params, other_quant_params, activation_quant_params, sepcial_list, prev_name=''):
        for name, child_module in module.named_children():
            tmp_name = prev_name+'_'+name
            if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)) and not ('downsample' in name and 'conv' in name):
                if check_in_special(tmp_name,sepcial_list):
                    setattr(module, name, self.IntLoRALayer(child_module, **lora_quant_params, activation_params=activation_quant_params))
                else:
                    setattr(module, name, QuantLayerNormal(child_module, other_quant_params,activation_params=activation_quant_params))
            elif isinstance(child_module, StraightThrough):
                continue
            else:
                self.quant_module_refactor(child_module, lora_quant_params, other_quant_params,activation_quant_params, sepcial_list, prev_name=tmp_name)


    def forward(self, image, t, context=None):
        return self.model(image, t, context)



