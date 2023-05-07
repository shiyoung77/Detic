import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Queue
from pathlib import Path
from itertools import repeat
from time import perf_counter


global available_devices


def init_process(_available_devices):
    global available_devices
    available_devices = _available_devices


def detic(dataset, video, idx, args):
    device = available_devices.get()
    print(f"Processing {video = }, {idx = }, {device = }")

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
    parser.add_argument("--dataset", type=str, default="~/t7/ScanNet/aligned_scans")
    parser.add_argument("--vocabulary", default="lvis")
    parser.add_argument("--confidence_thresh", type=float, default=0.3)
    parser.add_argument("--output_folder", default="detic_output")
    parser.add_argument("--num_gpus", type=int, default=8)
    args = parser.parse_args()

    dataset = Path(args.dataset).expanduser()
    videos = [i.name for i in sorted(dataset.iterdir())]

    _available_devices = Queue()
    for i in range(args.num_gpus):
        _available_devices.put(i)
    init_args = (_available_devices,)

    tic = perf_counter()
    with ProcessPoolExecutor(max_workers=args.num_gpus, initializer=init_process, initargs=init_args) as executor:
        executor.map(detic, repeat(dataset), videos, range(len(videos)), repeat(args))
    print(f"Process {len(videos)} takes {perf_counter() - tic}s")


if __name__ == "__main__":
    main()
