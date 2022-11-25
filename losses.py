import torch


def bin_mask_loss_multi(mask):
    return torch.minimum(mask, 1 - mask).sum(1).mean()


def min_mask_loss_multi(mask, min_mask_area):
    return torch.relu(min_mask_area - mask.mean((-1, -2))).sum(1).mean()


def max_mask_loss_multi(mask, max_mask_area):
    return torch.relu(mask.mean((-1, -2)) - max_mask_area).sum(1).mean()


class Loss:
    def __init__(self, key, loss_fn, alpha, name, warmup_iters=0):
        self.key = key
        self.loss_fn = loss_fn
        self.alpha = alpha
        self.name = name
        self.warmup_iters = warmup_iters
        self.iter = 0

    def __call__(self, return_dict):
        loss = self.loss_fn(return_dict[self.key])
        if self.iter < self.warmup_iters:
            alpha = self.alpha * self.iter / self.warmup_iters
            self.iter += 1
        else:
            alpha = self.alpha
        return {self.name: alpha * loss}, {self.name: loss}


class ComposeLoss:
    def __init__(self, *losses):
        self.losses = losses

    def __call__(self, return_dict):
        scaled_loss_dict, loss_dict = {}, {}
        for loss in self.losses:
            scaled_loss, loss = loss(return_dict)
            scaled_loss_dict.update(scaled_loss)
            loss_dict.update(loss)
        return scaled_loss_dict, loss_dict


class MultiLoss:
    def __init__(self, keys, loss_fn, alpha, name):
        self.keys = keys
        self.loss_fn = loss_fn
        self.alpha = alpha
        self.name = name

    def __call__(self, return_dict):
        loss = self.loss_fn(*[return_dict[key] for key in self.keys])
        return {self.name: self.alpha * loss}, {self.name: loss}
