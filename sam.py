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
from segment_anything import build_sam, SamPredictor


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
    parser.add_argument("-d", "--dataset", type=str, default="~/t7/ScanNet")
    parser.add_argument("-v", "--video", type=str, default="scene0011_00")
    parser.add_argument("--detic_exp", type=str, default="scan_net-0.3")
    parser.add_argument("--sam_checkpoint", type=str, default="~/Downloads/sam_vit_h_4b8939.pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=8)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    git_repo = Path(git.Repo(search_parent_directories=True).working_tree_dir)
    sys.path.append(str(git_repo))
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.chunk_id}"

    # initialize SAM
    predictor = SamPredictor(build_sam(checkpoint=Path(args.sam_checkpoint).expanduser()).to(args.device))

    dataset = Path(args.dataset).expanduser()
    video_folders = sorted((dataset / 'aligned_scans').iterdir())
    chunk_size = len(video_folders) // args.num_chunks + 1
    video_folders = video_folders[chunk_size * args.chunk_id: chunk_size * (args.chunk_id + 1)]
    print(f"{video_folders = }")

    for video_folder in video_folders:
        color_im_folder = video_folder / "color"
        detic_output_folder = video_folder / "detic_output" / args.detic_exp / "instances"

        for color_im_path in tqdm(sorted(color_im_folder.iterdir())):
            image = cv2.imread(str(color_im_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)

            detic_output_path = detic_output_folder / color_im_path.name.replace("jpg", "pkl")
            with open(detic_output_path, 'rb') as fp:
                instances = pickle.load(fp)

            boxes = instances.get("pred_boxes").tensor.to(args.device)  # XYXY
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image.shape[:2])

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

            output_path = detic_output_folder / color_im_path.name.replace(".jpg", "_sam.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump(instances, f)

            # draw output image
            # plt.figure(figsize=(10, 10))
            # for box, label, mask, quality, score in zip(boxes, pred_classes, masks, qualities, pred_scores):
            #     print(f"{quality = }, {score = }")
            #     plt.imshow(image)
            #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            #     show_box(box.cpu().numpy(), plt.gca(), f"{quality = }, {score = }")
            #     plt.show()


if __name__ == "__main__":
    main()
