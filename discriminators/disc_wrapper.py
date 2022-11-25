from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from discriminators.augment import AugmentPipe
from discriminators.diffaug import DiffAugment

from discriminators.pg_modules.discriminator import ProjectedDiscriminator
from utils_move import denormalize, normalize

class ADATracker:
    def __init__(self):
        self.n = torch.tensor(0.)
        self.n_signs = torch.tensor(0.)
        self.n_calls = 0
        self.bs = 0

    def update(self, real_pred):
        self.n_signs += torch.sign(real_pred).sum().cpu()
        self.n += torch.ones_like(real_pred).sum().cpu()
        self.n_calls += 1
        self.bs = real_pred.shape[0] # It should stay the same

    def __call__(self):
        return self.n_signs / self.n

    def reset(self):
        self.n = torch.tensor(0.)
        self.n_signs = torch.tensor(0.)
        self.n_calls = 0

class HingeLoss:

    def loss_d_real(self, output_dict, key='logits'):
        logits = output_dict[key]
        loss = (F.relu(torch.ones_like(logits) - logits)).mean()
        return {'d_real_loss': loss}
    
    def loss_d_fake(self, output_dict, key='logits'):
        logits = output_dict[key]
        loss = (F.relu(torch.ones_like(logits) + logits)).mean()
        return {'d_fake_loss': loss}

    def loss_d(self, output_dict_fake, output_dict_real, key='logits'):
        d1 = self.loss_d_fake(output_dict_fake, key)
        d1.update(self.loss_d_real(output_dict_real, key))
        return d1
    
    def loss_g(self, output_dict, key='logits'):
        logits = output_dict[key]
        loss = (-logits).mean()
        return {'g_loss': loss}

class DiscriminatorWrapper(nn.Module):
    def __init__(self, D, diffaug=False, diffaug_policy='color,translation',
        augment_p=0, ada_target=None, ada_interval=4, ada_kimg=500,
        gan_loss_fn=HingeLoss(),
        train_module=None,
        normalize_back=False):
        super().__init__()
        ada = augment_p>0 or ada_target is not None
        assert not (ada and diffaug)

        self.ada = ada
        self.ada_target = ada_target
        self.diffaug = diffaug
        self.diffaug_policy = diffaug_policy
        self.train_module = train_module

        self.gan_loss_fn = gan_loss_fn

        self.normalize_back = normalize_back

        self.D = D
        if ada:
            augment_kwargs = EasyDict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
            augment_pipe = AugmentPipe(**augment_kwargs).train().requires_grad_(False)
            augment_pipe.p.copy_(torch.as_tensor(augment_p))
            self.augment_pipe = augment_pipe
            self.ada_tracker = ADATracker()
            self.ada_interval = ada_interval # Careful; it might depend on the accum_iter
            self.ada_kimg = ada_kimg
            self.ada_last_stats = 0.
    
    def augment(self, x, renormalize=True, normalize_back=False):
        if renormalize:
            x = denormalize(x) * 2. - 1.
        if self.ada:
            x = self.augment_pipe(x)
        elif self.diffaug:
            x = DiffAugment(x, policy=self.diffaug_policy)
        if normalize_back:
            x = x * 0.5 + 0.5
            x = normalize(x)
        return x

    def run_D(self, x):
        x = self.augment(x, renormalize=True, normalize_back=self.normalize_back)
        return self.D(x)

    def loss_d(self, output_dicts_fake, output_dicts_real):
        log_dict = {}
        if self.ada and self.ada_target is not None:
            real_logits = torch.cat([v['logits'] for v in output_dicts_real.values()], dim=0)
            real_logits = self.train_module.all_gather(real_logits)
            real_logits = torch.cat([rl for rl in real_logits])
            self.ada_tracker.update(real_logits)
            if self.ada_tracker.n_calls % self.ada_interval == 0:
                # Adjust ADA probability
                ada_cur = self.ada_tracker()
                self.ada_last_stats = ada_cur
                adjust = torch.sign(ada_cur - self.ada_target) * (self.ada_tracker.bs * self.ada_interval) / (self.ada_kimg * 1000)
                new_p = min(max(self.augment_pipe.p.item() + adjust.item(), 0.), 1.)
                new_p = torch.tensor(new_p)
                self.augment_pipe.p.copy_(new_p)
                self.ada_tracker.reset()
            log_dict = {'ada_p': self.augment_pipe.p.item(), 'ada_stats': self.ada_last_stats}

        aux_loss = []
        for k, v in {**output_dicts_fake, **output_dicts_real}.items():
            al = v.pop('aux_loss', None)
            if al is not None:
                aux_loss.append(al)
        aux_loss = sum(aux_loss) if len(aux_loss) > 0 else None
        loss_dict = {}
        if aux_loss is not None:
            loss_dict['aux_loss'] = aux_loss

        fake_losses = {f'{k}_{k2}': self.gan_loss_fn.loss_d_fake(v, key=k2) for k, v in output_dicts_fake.items() for k2 in v.keys()}
        real_losses = {f'{k}_{k2}': self.gan_loss_fn.loss_d_real(v, key=k2) for k, v in output_dicts_real.items() for k2 in v.keys()}
        fake_losses = {f'{k1}_{k2}': v/len(output_dicts_fake) for k1, v1 in fake_losses.items() for k2, v in v1.items()}
        real_losses = {f'{k1}_{k2}': v/len(output_dicts_real) for k1, v1 in real_losses.items() for k2, v in v1.items()}

        loss_dict.update(fake_losses)
        loss_dict.update(real_losses)
        return loss_dict, log_dict

    def loss_g(self, output_dicts):
        log_dict = {}
        gen_losses = {f'{k}_{k2}': self.gan_loss_fn.loss_g(v, key=k2) for k, v in output_dicts.items() for k2 in v.keys()}
        gen_losses = {f'{k1}_{k2}': v/len(gen_losses) for k1, v1 in gen_losses.items() for k2, v in v1.items()}
        return gen_losses, log_dict


def build_discriminator(args, train_module):
    discriminator_ckpt = None if not hasattr(args, 'discriminator_ckpt') else args.discriminator_ckpt
    backbone_kwargs = dict(cout=64,
                            expand=True,
                            proj_type=2,
                            num_discs=4,
                            separable=False,
                            cond=False,
                            patch=False,
                            pretrained_ckpt=discriminator_ckpt,
                            )
    discriminator = ProjectedDiscriminator(backbone_kwargs=backbone_kwargs)
    gan_loss_fn = HingeLoss()

    diffaug = args.diffaug_policy is not None and args.diffaug_policy != ''
    discriminator_wrapper = DiscriminatorWrapper(discriminator,
                        diffaug=diffaug,
                        diffaug_policy=args.diffaug_policy,
                        # augment_p=args.augment_p,
                        # ada_target=args.ada_target,
                        # ada_interval=args.ada_interval,
                        # ada_kimg=args.ada_kimg,
                        gan_loss_fn=gan_loss_fn,
                        train_module=train_module,
                        normalize_back=discriminator_ckpt is not None)
    return discriminator_wrapper