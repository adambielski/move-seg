import os
import argparse
import numpy as np

from easydict import EasyDict
from tqdm import tqdm
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as F_tv
from PIL import Image

from train import TrainerLite
import bilateral_solver

device = torch.device("cuda")

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


class ImageFolderWPaths(ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        size = np.array(sample.size).tolist()
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path, size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--bilateral", action="store_true")
    parser.add_argument("--size", type=int, default=448)  # size of the shorter side
    parser.add_argument("--max_size", type=int, default=None)
    parser.add_argument(
        "--resize_to_square", action="store_true"
    )  # allows batching but loses aspect ratio
    parser.add_argument(
        "--bs", type=int, default=32
    )  # Set to 1 anyway unless resize_to_square
    parser.add_argument(
        "--out_dir", type=str, default=None
    )  # If None it will be put in the same folder as the model
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    ARGS_PATH = MODEL_PATH[:-4] + ".args"

    loaded = torch.load(ARGS_PATH)
    model_args = loaded["args"]

    if not args.resize_to_square:
        transform = [
            transforms.Resize(args.size, max_size=args.max_size, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
        transform = transforms.Compose(transform)
        args.bs = 1
    else:
        transform = [
            transforms.Resize((args.size, args.size), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
        transform = transforms.Compose(transform)

    db_folder = args.dataset
    imgset = ImageFolderWPaths(db_folder, transform=transform)
    all_paths_cat = [imgset.samples[idx][0] for idx in range(len(imgset))]
    dataloader = DataLoader(imgset, batch_size=args.bs, drop_last=False, shuffle=False)

    relpaths = ["/".join(p.split("/")[-2:]) for p in all_paths_cat]

    out_dir = args.out_dir
    if out_dir is None:
        db_name = db_folder.rstrip("/").split("/")[-1]
        size_suffix = f"_{args.size}"
        size_suffix += f"_maxsize{args.max_size}" if args.max_size is not None else ""
        size_suffix = "_square" if args.resize_to_square else (size_suffix + "_keepAR")
        out_dir = os.path.join(
            os.path.dirname(MODEL_PATH),
            "pred_"
            + db_name
            + size_suffix
            + "_"
            + os.path.basename(MODEL_PATH).split(".")[0],
        )

    os.makedirs(out_dir, exist_ok=True)

    outpaths = [os.path.join(out_dir, relpath) for relpath in relpaths]
    outpaths = [os.path.splitext(path)[0] + ".png" for path in outpaths]

    do_inference = False
    for outpath in outpaths:
        if ".ipynb_checkpoints" in outpath:
            continue
        if not os.path.exists(outpath):
            print(outpath)
            do_inference = True
            break

    if do_inference:
        print("LOADING MODEL")
        trainer = TrainerLite(
            model_args, None, strategy="dp", gpus=1, precision=model_args.precision
        )
        trainer.model = trainer.setup(trainer.model)
        ckpt = trainer.load(MODEL_PATH)

        trainer.model.load_state_dict(ckpt["model"], strict=False)

        segmenter = trainer.model.segmenter
        segmenter.eval()

        patch_size = (
            segmenter.feature_extractor.patch_size
            if segmenter.feature_extractor is not None
            else 16
        )
        all_masks = []
        all_paths = []
        with torch.no_grad():
            for data in tqdm(dataloader):
                imgs, _, paths = data[:3]
                if not args.resize_to_square:
                    h, w = torch.tensor(imgs.shape[-2:]).numpy()
                    imgs = F_tv.pad(
                        imgs,
                        padding_mode="reflect",
                        padding=(
                            0,
                            0,
                            (patch_size - w % patch_size) % patch_size,
                            (patch_size - h % patch_size) % patch_size,
                        ),
                    )  # pad for ViT

                masks = segmenter(imgs.cuda().float())

                if not args.resize_to_square:
                    masks = masks[:, :, :h, :w]

                all_masks.append(masks.cpu())
                all_paths.append(paths)

        all_masks_cat = []
        all_paths_cat = []
        for paths, masks in zip(all_paths, all_masks):
            all_paths_cat.extend(paths)
            all_masks_cat.extend([m.squeeze(0) for m in masks])

        relpaths = ["/".join(p.split("/")[-2:]) for p in all_paths_cat]
        outpaths = [os.path.join(out_dir, relpath) for relpath in relpaths]
        outpaths = [os.path.splitext(path)[0] + ".png" for path in outpaths]

        outdirs = set([os.path.dirname(path) for path in outpaths])
        for out_dir_ in outdirs:
            if ".ipynb_checkpoints" in out_dir_:
                continue
            print(out_dir_)
            os.makedirs(out_dir_, exist_ok=True)

        for mask, outpath in tqdm(
            zip(all_masks_cat, outpaths), total=len(all_masks_cat)
        ):
            if ".ipynb_checkpoints" in outpath:
                continue
            img = Image.fromarray((mask.numpy() * 255).astype(np.uint8), "L")
            img.save(outpath)

    all_masks = []
    for outpath in tqdm(outpaths):
        if ".ipynb_checkpoints" in outpath:
            continue
        img = np.array(Image.open(outpath)).astype(np.float32) / 255
        all_masks.append(img)

    if args.bilateral:
        print("Computing bilateral solver")

        bil_args = EasyDict()
        bil_args.sigma_spatial = 16
        bil_args.sigma_luma = 16
        bil_args.sigma_chroma = 8

        relpaths = ["/".join(p.split("/")[-2:]) for p in all_paths_cat]

        db_name_ = db_name + "-bilateral"
        out_dir_bilateral = out_dir + "_bilateral"
        os.makedirs(out_dir_bilateral, exist_ok=True)
        outpaths_bilateral = [
            os.path.join(out_dir_bilateral, relpath) for relpath in relpaths
        ]
        outpaths_bilateral = [
            os.path.splitext(path)[0] + ".png" for path in outpaths_bilateral
        ]

        bilateral_masks = []
        for path, mask, outpath in tqdm(
            zip(all_paths_cat, all_masks, outpaths_bilateral), total=len(all_masks)
        ):
            if os.path.exists(outpath):
                continue
            output_solver, binary_solver = bilateral_solver.bilateral_solver_output(
                path,
                mask,
                sigma_spatial=bil_args.sigma_spatial,
                sigma_luma=bil_args.sigma_luma,
                sigma_chroma=bil_args.sigma_chroma,
            )
            bilateral_masks.append(output_solver)

        outdirs = set([os.path.dirname(path) for path in outpaths_bilateral])
        for out_dir_ in outdirs:
            if ".ipynb_checkpoints" in out_dir_:
                continue
            print(out_dir_)
            os.makedirs(out_dir_, exist_ok=True)

        for mask, outpath in tqdm(
            zip(bilateral_masks, outpaths_bilateral), total=len(bilateral_masks)
        ):
            if ".ipynb_checkpoints" in outpath:
                continue
            if os.path.exists(outpath):
                continue
            img = Image.fromarray((mask * 255).astype(np.uint8), "L")
            img.save(outpath)
