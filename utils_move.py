import numpy as np
import torch
import torch.nn.functional as F


def translation(x, translation):
    translation_x = translation[:,1].view(-1,1,1)
    translation_y = translation[:,0].view(-1,1,1)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x

def translate_outputs(x, translations):
    x = torch.cat((x, torch.ones(x.shape[0], 1, *x.shape[2:], dtype=x.dtype, device=x.device)), dim=1)
    x = translation(x, -translations)
    x, masked = x[:,:-1], x[:,-1:]
    return x, masked

def shift(shifts, *tensors):
    dims = [t.shape[1] for t in tensors]
    cat_tensor = torch.cat(tensors, dim=1)
    x_shifted = translate_outputs(cat_tensor, shifts)[0]
    return x_shifted.split(dims, 1)

def unpatchify(x, patch_size, c=3, size=None):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = patch_size
        if size is None:
            h = w = int(x.shape[1]**.5)
            assert h * w == x.shape[1]
        else:
            h, w = size
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return x

def compute_masks(mask_org, mask_shifted):
    fg_mask = mask_shifted
    bg_mask = (1 - fg_mask) * (1 - mask_org)
    inpaint_mask = mask_org * (1 - fg_mask)
    real_mask = 1 - inpaint_mask
    return fg_mask, bg_mask, inpaint_mask, real_mask

class AvgLogDict:
    def __init__(self):
        self.dict_ = {}

    def __setitem__(self, key, value):
        if key in self.dict_:
            self.dict_[key].append(value)
        else:
            self.dict_[key] = [value]
    
    def __getitem__(self, key):
        tmp = self.dict_.__getitem__(key)
        return sum(tmp) / len(tmp)

class Normalizer:
    def __init__(self, mode='denormalize'):
        assert mode in ('denormalize', 'normalize')
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])

        self.std_tensor = torch.tensor(imagenet_std).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).float()
        self.mean_tensor = torch.tensor(imagenet_mean).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).float()
        self.mode = mode

    def __call__(self, x):
        if self.mode == 'denormalize':
            return x * self.std_tensor.to(x.device) + self.mean_tensor.to(x.device)
        else:
            return (x - self.mean_tensor.to(x.device)) / self.std_tensor.to(x.device)

denormalize = Normalizer(mode='denormalize')
normalize = Normalizer(mode='normalize')

class FnModifier:
    def __init__(self, key, modifier_fn, name):
        self.key = key
        self.modifier_fn = modifier_fn
        self.name = name
    
    def __call__(self, return_dict):
        return_dict[self.name] = self.modifier_fn(return_dict[self.key])
        return return_dict

class DictModifier:
    def __init__(self, modifier_fn, name):
        self.modifier_fn = modifier_fn
        self.name = name
    
    def __call__(self, return_dict):
        return_dict[self.name] = self.modifier_fn(return_dict)
        return return_dict

class ComposeModifier:
    def __init__(self, *modifiers):
        self.modifiers = modifiers
    
    def __call__(self, return_dict):
        for modifier in self.modifiers:
            return_dict = modifier(return_dict)
        return return_dict
