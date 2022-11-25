import os
import math
import urllib

import torch
import torch.nn as nn
import torch.nn.functional as F

import mae_models
from utils_move import shift, unpatchify, compute_masks

## Feature Extractor
class ViTFeatureExtractor(nn.Module):
    # -2 take the penultimate layer
    def __init__(self, model, block_layer=-2):
        super().__init__()
        model.requires_grad_(False)
        self.model = model  # We don't need to save the model
        if block_layer < 0:
            block_layer = len(model.blocks) + block_layer
        self.block_layer = block_layer
        self.n_blocks = len(model.blocks)

        self.patch_size = model.patch_embed.patch_size
        self.embed_dim = model.embed_dim

        if block_layer != len(model.blocks) - 1:
            self.norm = nn.LayerNorm(
                model.embed_dim, eps=1e-6
            )  # train the new norm layer if we don't use the last layer
        else:
            self.norm = model.norm

        self.train(True)

    def train(self, mode):
        super().train(mode)
        self.model.train(False)
        self.norm.train(mode and self.n_blocks != self.block_layer + 1)

    def forward(self, x):
        n, c, h, w = x.shape
        model = self.model
        x = model.prepare_tokens(x)
        for blk in model.blocks[: self.block_layer + 1]:
            x = blk(x)
        x = self.norm(x)
        x = x[:, 1:]  # remove CLS token
        ph, pw = h // self.patch_size, w // self.patch_size
        x = x.view(x.shape[0], ph, pw, x.shape[-1]).permute(0, 3, 1, 2)  # NCHW
        return x


def build_feature_extractor(args):
    model_name = args.feature_extractor  # default: 'dino_vits8'
    block_layer = args.extractor_block_layer  # default: -2 (penultimate layer)
    args.mae_as_feature_extractor = False
    if "dino" in model_name:
        extractor = torch.hub.load("facebookresearch/dino:main", model_name)
        extractor = ViTFeatureExtractor(extractor, block_layer=block_layer)
    elif "mae" in model_name:
        extractor = None  # MAE features will be computed as a byproduct of inpainting
        args.mae_as_feature_extractor = True
    else:
        raise ValueError("Unknown model name: {}".format(model_name))
    return extractor


## Segmenter head


def up_block(
    in_dim,
    out_dim,
    block_depth,
    kernel_size=3,
    padding_mode="zeros",
    no_upsampling=False,
):
    uplayer = nn.Upsample(scale_factor=(2, 2), mode="nearest")
    layers = [
        nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            padding_mode=padding_mode,
        ),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(),
    ]
    for i in range(block_depth - 1):
        layers += [
            nn.Conv2d(
                out_dim,
                out_dim,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(),
        ]
    if not no_upsampling:
        layers.insert(0, uplayer)
    return nn.Sequential(*layers)


class SegmenterConvHead(nn.Module):
    def __init__(
        self, upsampling_blocks, in_channels, widths, mask_channels, block_depth
    ):
        super().__init__()
        assert len(widths) == upsampling_blocks + 1
        layers = []
        widths = [in_channels] + widths

        for i in range(len(widths) - 1):
            layers.append(
                up_block(
                    widths[i],
                    widths[i + 1],
                    block_depth,
                    no_upsampling=(i == len(widths) - 2),
                )
            )
        layers.append(
            nn.Conv2d(widths[-1], mask_channels, kernel_size=1, stride=1, padding=0)
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x, return_log_imgs=False):
        x = self.model(x)
        # TODO: probably not needed
        if return_log_imgs:
            return x, {}
        return x


def find_argument(head_type, argument, default_value):
    if argument not in head_type:
        return default_value
    return head_type.split(argument)[-1].split("_")[0]


def build_segmenter_head(args, extractor):
    head_type = args.segmenter_head
    if extractor is not None:
        patch_size = extractor.patch_size
        embed_dim = extractor.embed_dim
    else:
        patch_size = 16  # MAE
        embed_dim = 1024  # MAE
    if head_type.startswith("conv"):
        upsampling_blocks = int(math.log2(patch_size))
        in_channels = embed_dim
        mask_channels = 1

        block_depth = int(find_argument(head_type, "blockd", 1))
        width_mult = float(find_argument(head_type, "wmult", 0.5))
        channel_min = int(find_argument(head_type, "minc", 64))

        widths = [
            max(channel_min, int(embed_dim * width_mult ** (i + 1)))
            for i in range(upsampling_blocks + 1)
        ]

        segmenter_head = SegmenterConvHead(
            upsampling_blocks=upsampling_blocks,
            in_channels=in_channels,
            widths=widths,
            mask_channels=mask_channels,
            block_depth=block_depth,
        )
        return segmenter_head
    else:
        raise NotImplementedError()


## Extractor-Segmenter wrapper


class ExtractorSegmenter(nn.Module):
    def __init__(self, feature_extractor, segmenter_head):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.segmenter_head = segmenter_head
        self.bin_fn = torch.sigmoid

    def forward(self, x, return_log_imgs=False):
        if len(x.shape) == 4:  # if NCHW images extract features
            x = self.feature_extractor(x)
        else:  # otherwise use the NTD (tokens x dim) features directly (e.g. for MAE features)
            side = int(math.sqrt(x.shape[1] - 1))
            x = (
                x[:, 1:].transpose(1, 2).view(x.shape[0], -1, side, side)
            )  # remove CLS token and reshape to NCHW
        x = self.segmenter_head(x, return_log_imgs=return_log_imgs)
        if not return_log_imgs:
            return self.bin_fn(x)
        else:
            return self.bin_fn(x[0]), x[1]


def build_segmenter(args):
    extractor = build_feature_extractor(args)
    segmenter_head = build_segmenter_head(args, extractor)
    segmenter = ExtractorSegmenter(extractor, segmenter_head)
    return segmenter


# MAE inpainting and composing


class MAEComposer(nn.Module):
    def __init__(
        self,
        mae_model,
        segmenter,
        shift_range=0.125,
        diff_inp_mask=True,
        mae_as_feature_extractor=False,
        copy_real_in_inpainted=False,
    ):

        super().__init__()
        mae_model.requires_grad_(False)
        mae_model.eval()
        self.mae_model = mae_model
        self.segmenter = segmenter
        self.train(True)

        self.shift_range = shift_range
        self.diff_inp_mask = diff_inp_mask

        self.copy_real_in_inpainted = copy_real_in_inpainted

        self.mae_patch_size = mae_model.patch_embed.patch_size[0]

        self.mae_as_feature_extractor = mae_as_feature_extractor

    def mask2tile(self, mask):
        """Converts a high-res NCHW mask to MAE's low-res mask indicating which tiles to use for inpainting"""
        return 1 - F.max_pool2d(
            mask, kernel_size=self.mae_patch_size, stride=self.mae_patch_size
        ).flatten(1)

    def train(self, mode):
        super().train(mode)
        self.mae_model.train(False)

    def interpolate_pos_encoding(self, x, w, h):
        mae_model = self.mae_model
        npatch = x.shape[1] - 1
        N = mae_model.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return mae_model.pos_embed
        # For different sizes, we interpolate the position encoding
        class_pos_embed = mae_model.pos_embed[:, 0]
        patch_pos_embed = mae_model.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // mae_model.patch_embed.patch_size[0]
        h0 = h // mae_model.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward_encoder_full(self, x):
        mae_model = self.mae_model
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            proj = mae_model.patch_embed.proj
            w, h = x.shape[
                -2:
            ]  # should be h, w but seems to be swapped in interpolate_pos_encoding
            x = proj(x).flatten(2).transpose(1, 2)
            pos_emb = self.interpolate_pos_encoding(x, w, h)
        else:
            x = mae_model.patch_embed(x)
            pos_emb = mae_model.pos_embed

        x = x + pos_emb[:, 1:, :]
        # append cls token
        cls_token = mae_model.cls_token + mae_model.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in mae_model.blocks:
            x = blk(x)
        x = mae_model.norm(x)

        return x

    def forward_decoder_full(self, x):
        mae_model = self.mae_model
        x = mae_model.decoder_embed(x)
        x = x + mae_model.decoder_pos_embed
        for blk in mae_model.decoder_blocks:
            x = blk(x)
        x = mae_model.decoder_norm(x)

        # predictor projection
        x = mae_model.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        return x

    def forward_decoder_soft(self, x, mask):
        # embed tokens
        mae_model = self.mae_model
        x = mae_model.decoder_embed(x)

        # Composition of MSK tokens and encoded tokens
        mask = mask.unsqueeze(-1)
        mask_tokens = mae_model.mask_token.repeat(x.shape[0], x.shape[1] - 1, 1)
        x_ = mask * x[:, 1:] + (1 - mask) * mask_tokens
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + mae_model.decoder_pos_embed

        # apply Transformer blocks
        for blk in mae_model.decoder_blocks:
            x = blk(x)
        x = mae_model.decoder_norm(x)

        # predictor projection
        x = mae_model.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        return x

    def forward_bg_inpaint_diff(self, x, bg_inp_mask, binarize=False):
        if not self.mae_as_feature_extractor:
            latent = self.forward_encoder_full(x)
        else:
            latent = x

        # if len(bg_inp_mask.shape) > 2:
        #     mask_patchified = self.mask2tile_fn(bg_inp_mask)
        # else:
        mask_patchified = bg_inp_mask
        if binarize:  # non-differentiable case
            mask_patchified = (mask_patchified > 0.5).float()
        pred = self.forward_decoder_soft(latent, mask_patchified)

        return pred, mask_patchified

    def ae(self, x):
        x = self.forward_encoder_full(x)
        x = self.forward_decoder_full(x)
        return unpatchify(x, self.mae_patch_size)

    def forward(self, x, shifts=None, return_log_imgs=False):
        """
        x: [B, C, H, W] image
        shifts: [B, 2] shift in x and y direction in pixels; if None: sampled uniformly from (-self.shift_range*H, self.shift_range*H) and (-self.shift_range*W, self.shift_range*W)
        """
        mae_model = self.mae_model
        if shifts is None:
            shifts = torch.rand(x.shape[0], 2, device=x.device) * 2 - 1
            shifts = shifts * self.shift_range
            side = min(x.shape[-2:])  # min(H, W) for now
            shifts = shifts * side
            shifts = shifts.round().long()

        # Predict the segmentation mask
        if self.mae_as_feature_extractor:
            latent = self.forward_encoder_full(
                x
            )  # used for both feature extraction & inpainting
        else:
            latent = x

        mask = self.segmenter(latent, return_log_imgs=return_log_imgs)

        if return_log_imgs:
            mask, log_img_dict = mask[:2]
        else:
            log_img_dict = {}

        # Random shift the image
        x_shifted, mask_shifted = shift(shifts, x, mask)

        # fg_mask - shifted mask; bg_mask - where the original background is in the composed image; inpaint_mask - where the inpainted part is in the composed image; real_mask: 1-inpaint_mask
        fg_mask, bg_mask, inpaint_mask, real_mask = compute_masks(mask, mask_shifted)

        bg_inp_mask = inpaint_mask + fg_mask

        # Inpaint the background
        # Convert high-res mask to MAE-patchified mask via max pooling
        bg_inp_mask_ = self.mask2tile(bg_inp_mask)

        pred_bg, bg_encoder_mask = self.forward_bg_inpaint_diff(
            latent, bg_inp_mask_, binarize=not self.diff_inp_mask
        )
        pred_bg = mae_model.unpatchify(pred_bg)

        if not self.copy_real_in_inpainted:
            # Autoencode before composing (default)
            x_ae_shifted = self.ae(x_shifted)
            x_ae = self.ae(x)
        else:
            # Option without autoencoding - the non-inpainted parts are copied from the original image
            inpainting_mask = 1 - F.interpolate(
                bg_encoder_mask.view(bg_encoder_mask.shape[0], 1, 14, 14),
                scale_factor=(16.0, 16.0),
            )  # Hardcoded for ViT/16 MAE
            pred_bg = pred_bg * inpainting_mask + x * (1 - inpainting_mask)
            x_ae_shifted = x_shifted
            x_ae = x

        # Compose the final image
        return_dict = {}
        composed = x_ae_shifted * fg_mask + pred_bg * (1 - fg_mask)

        return_dict.update(
            {
                "composed": composed,
                "pred_bg": pred_bg,
                "mask": mask,
                "mask_shifted": fg_mask,
                "real_ae_shifted": x_ae_shifted,
                "real_ae": x_ae,
                "bg_inp_mask": bg_inp_mask,
                "bg_encoder_mask": bg_encoder_mask,
            }
        )
        return_dict.update(log_img_dict)

        # Create the composed image without the shift
        composed_noshift = x_ae * mask + pred_bg * (1 - mask)
        return_dict["composed_noshift"] = composed_noshift
        return return_dict


MAE_MODELS = {
    "mae_nogan": {
        "arch": "mae_vit_large_patch16",
        "url": "https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth",
    },
    "mae_gan": {
        "arch": "mae_vit_large_patch16",
        "url": "https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth",
    },
}


def build_mae_model(mae_type):
    mae_model = MAE_MODELS[mae_type]
    url = mae_model["url"]
    mae_checkpoint = url.split("/")[-1]
    # Download a file if it doesn't exist
    if not os.path.exists(mae_checkpoint):
        print("Downloading model from {}".format(url))
        urllib.request.urlretrieve(url, mae_checkpoint)
    # Load the model
    model = getattr(mae_models, mae_model["arch"])()
    # load model
    checkpoint = torch.load(mae_checkpoint, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    print(msg)
    return model


def build_composer(args, segmenter):
    mae_as_feature_extractor = args.mae_as_feature_extractor
    mae_model = build_mae_model(args.mae_model)
    mae_model.requires_grad_(False)
    mae_model.eval()
    composer = MAEComposer(
        mae_model=mae_model,
        segmenter=segmenter,
        shift_range=args.shift_range,
        diff_inp_mask=not args.no_diff_inp_mask,
        mae_as_feature_extractor=mae_as_feature_extractor,
        copy_real_in_inpainted=args.copy_real_in_inpainted,
    )
    if mae_as_feature_extractor:
        segmenter.feature_extractor = MAEExtractorWrapper(composer.mae_model)

    return composer


# Used for inference if MAE is used as a feature extractor
class MAEExtractorWrapper(nn.Module):
    def __init__(self, mae_model):
        super().__init__()
        self.mae_model = mae_model
        self.patch_size = mae_model.patch_embed.patch_size[0]

    def interpolate_pos_encoding(self, x, w, h):
        model = self.mae_model
        npatch = x.shape[1] - 1
        N = model.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return model.pos_embed
        class_pos_embed = model.pos_embed[:, 0]
        patch_pos_embed = model.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // model.patch_embed.patch_size[0]
        h0 = h // model.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x):
        mae_model = self.mae_model
        h, w = x.shape[-2:]
        if h != 224 or w != 224:
            proj = mae_model.patch_embed.proj
            # norm = self.mae_model_gan.patch_embed.norm
            x = proj(x).flatten(2).transpose(1, 2)
            pos_emb = self.interpolate_pos_encoding(
                x, h, w
            )  # this has to be double checked
        else:
            x = mae_model.patch_embed(x)
            pos_emb = mae_model.pos_embed

        x = x + pos_emb[:, 1:, :]
        # append cls token
        cls_token = mae_model.cls_token + mae_model.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in mae_model.blocks:
            x = blk(x)
        x = mae_model.norm(x)

        x = x[:, 1:]  # remove CLS token
        ph, pw = h // self.patch_size, w // self.patch_size
        x = x.view(x.shape[0], ph, pw, x.shape[-1]).permute(0, 3, 1, 2)

        return x
