import os
import sys
import argparse
import pickle
from pathlib import Path

import git
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
from tqdm import tqdm
from segment_anything import build_sam, build_sam_vit_b, SamPredictor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def masks_to_rle(masks):
    """
    https://github.com/facebookresearch/detectron2/issues/347
    """
    pred_masks_rle = []
    for mask in masks:
        rle = mask_util.encode(np.asfortranarray(mask))
        rle['counts'] = rle['counts'].decode('utf-8')
        pred_masks_rle.append(rle)
    return pred_masks_rle


def rle_to_masks(instances):
    instances.pred_masks = torch.from_numpy(
        np.stack([mask_util.decode(rle) for rle in instances.pred_masks_rle]))
    return instances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="~/t7/ScanNet/aligned_scans")
    parser.add_argument("--detic_exp", type=str, default="imagenet21k-0.3")
    parser.add_argument("--video", default="scene0011_00")
    parser.add_argument("--sam_checkpoint", type=str, default="~/Downloads/sam_vit_h_4b8939.pth")
    # parser.add_argument("--sam_checkpoint", type=str, default="~/Downloads/sam_vit_b_01ec64.pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("stride", type=int, default=5, help="stride of frames to process")
    args = parser.parse_args()

    # initialize SAM
    # predictor = SamPredictor(build_sam_vit_b(checkpoint=Path(args.sam_checkpoint).expanduser()).to(args.device))
    predictor = SamPredictor(build_sam(checkpoint=Path(args.sam_checkpoint).expanduser()).to(args.device))

    dataset = Path(args.dataset).expanduser()

    color_im_folder = dataset / args.video / "color"
    detic_output_folder = dataset / args.video / "detic_output" / args.detic_exp / "instances"
    sam_output_folder = dataset / args.video / "sam_output" / args.detic_exp / "instances"

    color_im_paths = sorted(color_im_folder.iterdir())

    filtered_color_im_paths = []
    for i in range(0, len(color_im_paths), args.stride):
        filtered_color_im_paths.append(color_im_paths[i])

    for color_im_path in tqdm(filtered_color_im_paths, disable=False):
        image = cv2.imread(str(color_im_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        detic_output_path = detic_output_folder / color_im_path.name.replace("jpg", "pkl")
        with open(detic_output_path, 'rb') as fp:
            instances = pickle.load(fp)

        boxes = instances.get("pred_boxes").tensor.to(args.device)  # XYXY
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
        if transformed_boxes.shape[0] != 0:
            masks, qualities, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            masks = masks.detach().squeeze().cpu().numpy().astype(np.bool_)
            qualities = qualities.squeeze().detach().cpu().numpy()
            instances.sam_masks_rle = masks_to_rle(masks)
            instances.sam_qualities = qualities
        else:
            instances.sam_masks_rle = []
            instances.sam_qualities = []

        output_path = sam_output_folder / color_im_path.name.replace(".jpg", ".pkl")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(instances, f)

        # draw output image
        # plt.figure(figsize=(10, 10))
        # for box, mask, quality in zip(boxes, masks, qualities):
        #     print(f"{quality = }")
        #     plt.imshow(image)
        #     show_mask(mask, plt.gca(), random_color=True)
        #     show_box(box.cpu().numpy(), plt.gca(), f"{quality = }")
        #     plt.show()
        # break


if __name__ == "__main__":
    main()
