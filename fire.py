import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Queue
from pathlib import Path
from itertools import repeat
from time import perf_counter

import torch


global available_devices


def init_process(_available_devices):
    global available_devices
    available_devices = _available_devices


def detic(dataset, video, idx, args):
    device = available_devices.get()
    print(f"Processing {video = }, {idx = }, {device = }")

    if args.save_vis:  # very slow
        command = f"""
            OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES={device} python demo_custom.py \
                --config-file "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml" \
                --dataset "{dataset}" \
                --video "{video}" \
                --output_folder "{args.output_folder}" \
                --vocabulary "{args.vocabulary}" \
                --confidence-threshold {args.confidence_thresh} \
                --save_vis \
                --opts MODEL.WEIGHTS "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
        """
    else:
        command = f"""
            OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES={device} python demo_custom.py \
                --config-file "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml" \
                --dataset "{dataset}" \
                --video "{video}" \
                --output_folder "{args.output_folder}" \
                --vocabulary "{args.vocabulary}" \
                --confidence-threshold {args.confidence_thresh} \
                --opts MODEL.WEIGHTS "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
        """
    subprocess.run(command, shell=True)
    available_devices.put(device)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, default="~/dataset/ScanNet/aligned_scans")
    parser.add_argument("--dataset", type=str, default="~/t7/ScanNet/aligned_scans")
    # parser.add_argument("--dataset", type=str, default="~/t7/ycb_video")
    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--vocabulary", default="imagenet21k")
    parser.add_argument("--confidence_thresh", type=float, default=0.3)
    parser.add_argument("--output_folder", default="detic_output")
    parser.add_argument("--save_vis", action='store_true')
    parser.add_argument("--num_gpus", type=int, default=-1, help="-1 means using all the visible gpus")
    args = parser.parse_args()

    if args.num_gpus == -1:
        max_workers = torch.cuda.device_count()
    else:
        max_workers = min(args.num_gpus, torch.cuda.device_count())

    dataset = Path(args.dataset).expanduser()
    if "ycb_video" in args.dataset:
        videos = [f"{i:04d}" for i in range(48, 60)]
    else:
        videos = [i.name for i in sorted(dataset.iterdir())]
    if args.video:
        videos = [args.video]

    print(f"{dataset = }")
    print(f"{len(videos) = }")
    print(f"{max_workers = }")

    _available_devices = Queue()
    for i in range(max_workers):
        _available_devices.put(i)
    init_args = (_available_devices,)

    tic = perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_process, initargs=init_args) as executor:
        executor.map(detic, repeat(dataset), videos, range(len(videos)), repeat(args))
    print(f"Process {len(videos)} takes {perf_counter() - tic}s")


if __name__ == "__main__":
    main()
