import argparse
import os
import sys
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms.functional as F


def IoU(mask1, mask2):
    mask1, mask2 = mask1.to(torch.bool), mask2.to(torch.bool)
    intersection = torch.sum(mask1 * (mask1 == mask2), dim=[-1, -2]).squeeze()
    union = torch.sum(mask1 + mask2, dim=[-1, -2]).squeeze()
    return (intersection.to(torch.float) / (union + 1e-8)).mean().item()


def accuracy(mask1, mask2):
    mask1, mask2 = mask1.to(torch.bool), mask2.to(torch.bool)
    return torch.mean((mask1 == mask2).to(torch.float)).item()


def precision_recall(mask_gt, mask):
    mask_gt, mask = mask_gt.to(torch.bool), mask.to(torch.bool)
    true_positive = torch.sum(mask_gt * (mask_gt == mask), dim=[-1, -2]).squeeze()
    mask_area = torch.sum(mask, dim=[-1, -2]).to(torch.float)
    mask_gt_area = torch.sum(mask_gt, dim=[-1, -2]).to(torch.float)

    precision = true_positive / mask_area
    precision[mask_area == 0.0] = 1.0

    recall = true_positive / mask_gt_area
    recall[mask_gt_area == 0.0] = 1.0

    return precision.item(), recall.item()


def F_score(p, r, betta_sq=0.3):
    f_scores = ((1 + betta_sq) * p * r) / (betta_sq * p + r)
    f_scores[f_scores != f_scores] = 0.0  # handle nans
    return f_scores


def F_max(precisions, recalls, betta_sq=0.3):
    F = F_score(precisions, recalls, betta_sq)
    F = F.mean(dim=0)
    F_argmax = F.argmax().item()
    F_max = F[F_argmax].item()

    return F_max, F_argmax


import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_dir")
    parser.add_argument("pred_dir")
    parser.add_argument("--th", default=0.5, type=float)
    parser.add_argument("--fbeta", action="store_true", default=False)
    parser.add_argument("--out", default=None)

    args = parser.parse_args()

    if args.out is None:
        args.out = f"metrics.txt"
        if args.fbeta:
            args.out = "extended_" + args.out

    subdirs = [os.path.join(args.gt_dir, d) for d in os.listdir(args.gt_dir)]
    subdirs = [d for d in subdirs if os.path.isdir(d)]
    gt_files = [
        os.path.join(subdir, p) for subdir in subdirs for p in os.listdir(subdir)
    ]
    gt_files = [f for f in gt_files if f.lower().endswith(".png")]

    pred_files = [os.path.join(args.pred_dir, f[len(args.gt_dir) :]) for f in gt_files]

    ious = []
    accuracies = []

    th = args.th
    resize_type = Image.BILINEAR

    precisions = []
    recalls = []
    prob_bins = 256

    for gt_file, pred_file in tqdm(zip(gt_files, pred_files), total=len(gt_files)):
        gt_mask = Image.open(gt_file)
        pred_mask = Image.open(pred_file).convert("L")
        pred_mask = pred_mask.resize(gt_mask.size, resize_type)

        gt_mask = torch.from_numpy(np.array(gt_mask).astype(np.float32)) / 255
        pred_mask = torch.from_numpy(np.array(pred_mask).astype(np.float32)) / 255

        if len(gt_mask.shape) > 2:
            print(f"warn: {gt_mask.shape} channels in {gt_file}")
            gt_mask = gt_mask[:, :, 0]

        gt_mask = (gt_mask > 0.5).float()

        # F Beta code
        if args.fbeta:
            p, r = [], []
            acc1, iou1 = [], []
            mae1 = []
            mae_absolute1 = []
            splits = (
                2.0 * pred_mask.mean(dim=0)
                if prob_bins is None
                else np.arange(0.0, 1.0, 1.0 / prob_bins)
            )

            for split in splits:
                pr = precision_recall(gt_mask, pred_mask > split)
                p.append(pr[0])
                r.append(pr[1])

            precisions.append(p)
            recalls.append(r)

            pred_mask = (pred_mask > th).float()

            ious.append(IoU(gt_mask, pred_mask))
            accuracies.append(accuracy(gt_mask, pred_mask))

        else:
            pred_mask = (pred_mask > th).float()
            ious.append(IoU(gt_mask, pred_mask))
            accuracies.append(accuracy(gt_mask, pred_mask))
            precision, recall = precision_recall(gt_mask, pred_mask)
            precisions.append([precision])
            recalls.append([recall])

    if args.fbeta:
        F_beta_max, F_beta_argmax = F_max(
            torch.tensor(precisions), torch.tensor(recalls)
        )
        fbetamaxes = []
        for pr, rc in zip(torch.tensor(precisions), torch.tensor(recalls)):
            fbm, _ = F_max(pr.unsqueeze(0), rc.unsqueeze(0))
            fbetamaxes.append(fbm)
        F_beta_max_perimg = torch.tensor(fbetamaxes).mean()

        half_th = args.th
        th_ious = np.array(ious)
        half_iou = np.array(ious).mean()
        half_acc = np.array(accuracies).mean()

        out = ""
        out += f"Threshold: {half_th}\n"
        out += f"Mean IoU: {half_iou}\n"
        out += f"Mean Accuracy: {half_acc}\n"

        out += f"FBetaMax: {F_beta_max}\n"
        out += f"FBetaArgMax: {F_beta_argmax}\n"
        out += f"FBetaMaxPerImg: {F_beta_max_perimg}\n"
    else:
        th_ious = np.array(ious)
        mean_iou = np.array(ious).mean()
        mean_acc = np.array(accuracies).mean()
        f_beta = F_max(torch.tensor(precisions), torch.tensor(recalls))

        out = ""
        out += f"Threshold: {th}\n"
        out += f"Mean IoU: {mean_iou}\n"
        out += f"Mean Accuracy: {mean_acc}\n"
        out += f"FBeta (not max): {f_beta}\n"

    print(out)

    with open(os.path.join(args.pred_dir, args.out), "w") as f:
        f.write(out)

    with open(os.path.join(args.pred_dir, "ious_" + args.out), "w") as f:
        for pred_file, iou in zip(pred_files, th_ious):
            f.write("/".join(pred_file.split("/")[-2:]) + f" {iou}\n")

    with open(os.path.join(args.pred_dir, "command_eval_" + args.out), "w") as f:
        f.write(" ".join(sys.argv))
