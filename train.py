import argparse
from functools import partial
from pathlib import Path
from tqdm import tqdm
import os
import torch
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image

from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.lite import LightningLite

from losses import (
    ComposeLoss,
    Loss,
    bin_mask_loss_multi,
    max_mask_loss_multi,
    min_mask_loss_multi,
)
from segmenter import build_segmenter, build_composer
from callbacks import ModelSaver, filter_requires_grad_parameters, get_imgs_to_log
from utils_move import ComposeModifier, DictModifier, FnModifier

from engine import train_one_epoch
from discriminators.disc_wrapper import build_discriminator
from utils_move import denormalize


def build_datasets(data_path, transform_train):
    """
    Get the datasets from the data_path
    Possible concatenation of datasets, separated by comma
    """
    data_paths = data_path.split(",")
    if len(data_paths) == 1:
        data_paths = [data_paths[0]]
    datasets = []
    for data_path in data_paths:
        dataset_train = tv.datasets.ImageFolder(data_path, transform=transform_train)
        datasets.append(dataset_train)
    if len(datasets) == 1:
        dataset_train = datasets[0]
    else:
        dataset_train = torch.utils.data.ConcatDataset(datasets)

    return dataset_train


def build_dataloaders(args, dataset_train):
    """
    Get the dataloaders from the dataset_train
    """
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=args.pin_mem,
        num_workers=args.num_workers,
        drop_last=True,
    )
    return dataloader_train


def build_modifiers(args):
    modifiers = []
    if "real_ae_shift_cp" in args.disc_fake_input:
        # fn for creating copy-paste fakes with predicted masks and real images
        realcp_fn = lambda rd: rd["real_ae_shifted"] * rd["mask_shifted"] + rd[
            "real_ae"
        ] * (1 - rd["mask_shifted"])
        modifiers.append(DictModifier(realcp_fn, "real_ae_shift_cp"))
    return ComposeModifier(*modifiers)


def build_optimizers(args, segmenter, discriminator):
    betas = tuple([float(b) for b in args.betas.split(",")])
    optimizer = torch.optim.AdamW(
        filter_requires_grad_parameters(segmenter.parameters()),
        lr=args.lr,
        betas=betas,
        weight_decay=args.wd,
    )
    betas_d = tuple([float(b) for b in args.betas_d.split(",")])
    optimizer_d = torch.optim.AdamW(
        filter_requires_grad_parameters(discriminator.parameters()),
        lr=args.lr_d,
        betas=betas_d,
    )
    return optimizer, optimizer_d


def build_mask_losses(args):
    """
    Build loss functions for mask binarization, minimum mask etc.
    Processes the composer output dictionary to create extra outputs (e.g. downsampled masks) and returns loss functions for some of the outputs
    Returns:
    - modifiers, that takes a dictionary of outputs from the composer model and adds modified outputs used for loss calculation
    - loss function, that takes a dictionary of outputs from the composer model and calculates the losses
    """
    modifiers = []
    downsamples = args.downsampled_mask_losses.split(",")
    mask_key = "mask" if args.mask_loss_on == "org" else "mask_shifted"
    mask_keys = [mask_key]
    if len(downsamples) > 0:
        for downsample in downsamples:
            if downsample == "avg":
                modifiers.append(
                    FnModifier(
                        key=mask_key,
                        modifier_fn=lambda x: F.avg_pool2d(
                            x, kernel_size=16, stride=16
                        ),
                        name="mask_avg16",
                    )
                )
            elif downsample == "max":
                modifiers.append(
                    FnModifier(
                        key=mask_key,
                        modifier_fn=lambda x: F.max_pool2d(
                            x, kernel_size=16, stride=16
                        ),
                        name="mask_max16",
                    )
                )
    mask_keys += [mod.name for mod in modifiers]

    loss_fns = [
        bin_mask_loss_multi,
        partial(min_mask_loss_multi, min_mask_area=args.min_mask_area),
        partial(max_mask_loss_multi, max_mask_area=args.max_mask_area),
    ]
    prefixes = ["bin", "minarea", "maxarea"]
    alphas = [args.bin_mask_alpha, args.min_mask_alpha, args.max_mask_alpha]
    warmup_iters = [args.bin_warmup_iters, 0, 0]
    losses = []
    for key in mask_keys:
        for loss_fn, prefix, alpha, warmup in zip(
            loss_fns, prefixes, alphas, warmup_iters
        ):
            losses.append(
                Loss(
                    key=key,
                    loss_fn=loss_fn,
                    alpha=alpha,
                    name=f"{prefix}_{key}",
                    warmup_iters=warmup,
                )
            )

    compose_modifier = ComposeModifier(*modifiers)
    compose_loss = ComposeLoss(*losses)

    return compose_modifier, compose_loss


class TrainerLite(LightningLite):
    def __init__(self, args, train_dataset, no_print=False, *vargs, **kwargs):
        super().__init__(*vargs, **kwargs)
        self.print("Seed is {}".format(args.seed))

        self.seed_everything(args.seed)
        if train_dataset:
            self.dataset = train_dataset
            # Optional images for logging
            self.imgs_to_log = get_imgs_to_log(self.dataset, args.n_imgs_to_log)
            self.print("Building dataloaders")
            self.dataloader = build_dataloaders(args, self.dataset)

        self.print("Building segmenter")
        segmenter = build_segmenter(args)

        self.print("Building composer")
        composer = build_composer(args, segmenter)
        if not no_print:
            self.print(composer)
        self.model = composer

        if train_dataset:
            self.print("Building discriminator")
            discriminator = build_discriminator(args, self)
            self.print(discriminator)
            self.discriminator = discriminator

            self.print("Building mask binarization/min mask losses")
            self.gen_modifiers, self.losses = build_mask_losses(args)

            # Build modifiers to create some of the discriminator inputs, e.g. shifted copy-paste fake inputs
            self.print("Building modifiers")
            self.disc_modifiers = build_modifiers(args)

            # Build optimizers
            self.print("Building optimizers")
            self.optimizer, self.optimizer_d = build_optimizers(
                args, segmenter, discriminator
            )

    def run(self, args):
        # Build loggers and callbacks
        dataloader = self.setup_dataloaders(self.dataloader)
        saver = ModelSaver()
        self.model, self.optimizer = self.setup(self.model, self.optimizer)
        self.discriminator, self.optimizer_d = self.setup(
            self.discriminator, self.optimizer_d
        )

        if self.is_global_zero:
            # Set up loggers that you want to use
            imgs_to_log = self.imgs_to_log
            # imgs_to_log_grid = make_grid(denormalize(imgs_to_log), nrow=int(imgs_to_log.shape[0]**0.5))
            fpath = os.path.join(args.output_dir, "imgs_to_log.png")
            save_image(
                denormalize(imgs_to_log), fpath, nrow=int(imgs_to_log.shape[0] ** 0.5)
            )
            with torch.no_grad():
                return_dict = self.model(
                    self.imgs_to_log.to(self.device), return_log_imgs=True
                )
                # Save the masks or intermediate outputs you want to log
                fpath = os.path.join(args.output_dir, "masks_init.png")
                save_image(
                    return_dict["mask"], fpath, nrow=int(imgs_to_log.shape[0] ** 0.5)
                )

        if args.resume:
            saver.load(
                args,
                self,
                self.model,
                self.optimizer,
                self.discriminator,
                self.optimizer_d,
            )
        else:
            saver.save(
                args,
                0,
                self,
                self.model,
                self.optimizer,
                self.discriminator,
                self.optimizer_d,
            )

        # Train
        for epoch in range(args.start_epoch, args.epochs):
            global_iter = epoch * len(dataloader)

            train_one_epoch(
                args,
                data_loader=dataloader,
                model=self.model,
                discriminator=self.discriminator,
                optimizer=self.optimizer,
                optimizer_d=self.optimizer_d,
                epoch=epoch,
                modifiers_gen=self.gen_modifiers,
                modifiers_disc=self.disc_modifiers,
                losses=self.losses,
                train_module=self,
                gen_keys=["composed"],
                fake_keys=args.disc_fake_input.split(
                    ","
                ),  # outputs used as fake input to train the discriminator
                real_keys=args.disc_real_input.split(
                    ","
                ),  # outputs used as real input to train the discriminator
                accum_iter=args.accum_iter,
                log_every=args.log_every,
            )

            # Save some images (customize as you want, might be useful in train_one_epoch instead)
            if self.is_global_zero:
                with torch.no_grad():
                    return_dict = self.model(
                        self.imgs_to_log.to(self.device), return_log_imgs=True
                    )
                    # Save the masks or intermediate outputs you want to log
                    fpath = os.path.join(
                        args.output_dir, f"masks_epoch_{epoch:04d}.png"
                    )
                    save_image(
                        return_dict["mask"],
                        fpath,
                        nrow=int(imgs_to_log.shape[0] ** 0.5),
                    )

            self.barrier()

            if args.output_dir and (
                epoch % args.save_every_epoch == 0 or epoch == args.epochs - 1
            ):
                saver.save(
                    args,
                    epoch+1,
                    self,
                    self.model,
                    self.optimizer,
                    self.discriminator,
                    self.optimizer_d,
                )
            self.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data specific
    parser.add_argument("--data_path", help="data directory", required=True)
    parser.add_argument(
        "--input_size", default=224, type=int, help="images input size"
    )  # Warning: sizes other than 224 are not tested

    # small data augmentation
    parser.add_argument("--scale_min", type=float, default=0.9)
    parser.add_argument("--scale_max", type=float, default=1.0)

    # Segmenter
    parser.add_argument("--segmenter_head", default="conv_minc128")
    parser.add_argument(
        "--feature_extractor", default="dino_vits8", choices=["dino_vits8", "mae"]
    )  # Other variants possible, edit to allow e.g. different DINO models from torch hub
    parser.add_argument(
        "--extractor_block_layer",
        default=-2,
        type=int,
        help="block layer to extract features from",
    )

    # Composer
    parser.add_argument("--no_diff_inp_mask", action="store_true")
    parser.add_argument("--shift_range", type=float, default=0.125)
    parser.add_argument(
        "--mae_model", default="mae_gan", choices=["mae_gan", "mae_nogan"]
    )
    parser.add_argument("--copy_real_in_inpainted", action="store_true")

    # Losses
    parser.add_argument(
        "--downsampled_mask_losses",
        default="avg,max",
        choices=["", "avg", "max", "avg,max"],
    )
    parser.add_argument("--min_mask_alpha", type=float, default=100.0)
    parser.add_argument("--max_mask_alpha", type=float, default=0.0)
    parser.add_argument("--min_mask_area", type=float, default=0.05)
    parser.add_argument("--max_mask_area", type=float, default=1.0)
    parser.add_argument("--bin_mask_alpha", type=float, default=12.5)
    parser.add_argument("--avg_bin_mask_alpha", type=float, default=None)

    parser.add_argument("--mask_loss_on", default="shifted", choices=["org", "shifted"])
    parser.add_argument("--bin_warmup_iters", type=int, default=2500)

    # Discriminator
    parser.add_argument(
        "--disc_real_input", default="real_ae,composed_noshift"
    )  ## TODO: the option for not AE
    parser.add_argument("--disc_fake_input", default="composed,real_ae_shift_cp")
    parser.add_argument("--diffaug_policy", default="color")

    # Optimization
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_d", type=float, default=2e-4)
    parser.add_argument("--betas", default="0.9,0.95")
    parser.add_argument("--betas_d", default="0.,0.99")
    parser.add_argument("--wd", type=float, default=0.0)

    # Training
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--resume", default=None)  # path to a checkpoint
    parser.add_argument("--accum_iter", default=1, type=int)

    # Logging
    parser.add_argument("--output_dir", default="output/move1")
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--n_imgs_to_log", default=16, type=int)
    parser.add_argument("--log_img_every", default=2500, type=int)

    parser.add_argument("--save_every_epoch", type=int, default=5)

    # Lightning lite
    parser.add_argument("--strategy", default="ddp")
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--precision", default=32, type=int)

    args = parser.parse_args()
    args.avg_bin_mask_alpha = (
        args.avg_bin_mask_alpha if args.avg_bin_mask_alpha else args.bin_mask_alpha
    )

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    args.log_dir = args.output_dir
    args.name = args.output_dir.split("/")[-1]

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                args.input_size, scale=(args.scale_min, args.scale_max), interpolation=3
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = build_datasets(args.data_path, transform_train)

    trainer = TrainerLite(
        args, dataset, strategy=args.strategy, gpus=args.gpus, precision=args.precision
    )
    trainer.run(args)
