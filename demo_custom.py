# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
import sys
import pickle

import torch
import cv2
import numpy as np
import pycocotools.mask as mask_util
from tqdm import tqdm

from detectron2.structures import Instances
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.predictor import VisualizationDemo


def setup_cfg(args):
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'  # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument("--config-file", default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml")
    parser.add_argument("--dataset", default="/home/lsy/dataset/CoRL_real")
    parser.add_argument("--video", default="0001")
    parser.add_argument("--output_folder", default="detic_output")
    parser.add_argument("--vocabulary", default="lvis", choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'])
    parser.add_argument("--custom_vocabulary", default="", help="comma separated words")
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--opts", help="'KEY VALUE' pairs", default=[], nargs=argparse.REMAINDER)
    return parser


def masks_to_rle(instances: Instances):
    """
    https://github.com/facebookresearch/detectron2/issues/347
    """
    pred_masks_rle = []
    for mask in instances.pred_masks:
        rle = mask_util.encode(np.asfortranarray(mask.numpy()))
        rle['counts'] = rle['counts'].decode('utf-8')
        pred_masks_rle.append(rle)
    instances.pred_masks_rle = pred_masks_rle
    return instances


def rle_to_masks(instances: Instances):
    instances.pred_masks = torch.from_numpy(np.stack([mask_util.decode(rle) for rle in instances.pred_masks_rle]))
    return instances


def main():
    # mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg, args)

    # for name, submodule in demo.predictor.model.named_children():  # [backbone, proposal_generator, roi_heads]
    #     print("================")
    #     print(name)
    #     for layer_name, layer in submodule.named_children():
    #         print(f"\t{layer_name}")

    metadata = MetadataCatalog.get("lvis_v1_val")

    video_path = os.path.join(args.dataset, args.video)
    rgb_files = sorted(os.listdir(os.path.join(video_path, 'color')))
    output_folder = os.path.join(args.dataset, args.video, args.output_folder)
    os.makedirs(os.path.join(output_folder, 'vis_imgs'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'instances'), exist_ok=True)

    for rgb_file in tqdm(rgb_files):
        rgb_path = os.path.join(video_path, 'color', rgb_file)
        img = read_image(rgb_path, format="BGR")
        instances = demo.predict_instances_only(img)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        visualizer = Visualizer(img_rgb, metadata, instance_mode=ColorMode.IMAGE)
        vis_output = visualizer.draw_instance_predictions(predictions=instances)
        vis_im = vis_output.get_image()
        output_path = os.path.join(output_folder, 'vis_imgs', f"{os.path.splitext(rgb_file)[0]}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR))

        masks_to_rle(instances)
        instances.remove('pred_masks')
        output_path = os.path.join(output_folder, 'instances', f"{os.path.splitext(rgb_file)[0]}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(instances, f)

        # with open(output_path, 'rb') as f:
        #     instances = pickle.load(f)
        #     instances = rle_to_masks(instances)


if __name__ == "__main__":
    main()
