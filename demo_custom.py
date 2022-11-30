# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
import sys
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

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
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--opts", help="'KEY VALUE' pairs", default=[], nargs=argparse.REMAINDER)
    return parser


def main():
    # mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg, args)

    video_path = os.path.join(args.dataset, args.video)
    rgb_files = sorted(os.listdir(os.path.join(video_path, 'color')))
    output_folder = os.path.join(args.dataset, args.video, args.output_folder)
    os.makedirs(output_folder, exist_ok=True)

    for rgb_file in tqdm(rgb_files):
        rgb_path = os.path.join(video_path, 'color', rgb_file)
        img = read_image(rgb_path, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)
        print(predictions)
        output_path = os.path.join(output_folder, rgb_file)
        visualized_output.save(output_path)
        break


if __name__ == "__main__":
    main()
