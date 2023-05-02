import subprocess
import concurrent.futures
from pathlib import Path


def detic(video, idx):
    DATASET = Path("~/dataset/ScanNet/aligned_scans").expanduser()
    VOCABULARY = "lvis"
    CONFIDENCE_THRESH = 0.3

    NUM_GPUS = 8
    DEVICE = idx % NUM_GPUS
    print(f"Processing {video = }, {idx = }, {DEVICE = }")

    command = f"""
        CUDA_VISIBLE_DEVICES={DEVICE} python demo_custom.py \
            --config-file "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml" \
            --dataset "{DATASET}" \
            --video "{video}" \
            --output_folder "detic_output" \
            --vocabulary "{VOCABULARY}" \
            --confidence-threshold {CONFIDENCE_THRESH} \
            --opts MODEL.WEIGHTS "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    """
    subprocess.run(command, shell=True)


def proposed_fusion(video, idx):
    DATASET = Path("~/t7/ScanNet").expanduser()
    IOU_THRESH = 0.3
    RECALL_THRESH = 0.4
    DEPTH_THRESH = 0.04
    FEATURE_SIMILARITY_THRESH = 0.7
    SIZE_THRESH = 100
    NUM_GPUS = 8
    DEVICE = f"cuda:{idx % NUM_GPUS}"
    print(f"Processing {video = }, {idx = }, {DEVICE = }")

    command = f"""
        python proposed_fusion.py \
            --dataset "{DATASET}" \
            --video "{video}" \
            --detic "scan_net-0.3" \
            --iou_thresh "{IOU_THRESH}" \
            --recall_thresh "{RECALL_THRESH}" \
            --feature_similarity_thresh "{FEATURE_SIMILARITY_THRESH}" \
            --depth_thresh "{DEPTH_THRESH}" \
            --size_thresh "{SIZE_THRESH}" \
            --device "{DEVICE}" \
            --vocab_feature_file "src/scannet200.npy" \
    """
    subprocess.run(command, shell=True)


def main():
    dataset = Path("~/t7/ScanNet/aligned_scans").expanduser()
    # videos = [i.name for i in sorted(dataset.iterdir())]
    videos = [i.name for i in sorted(dataset.iterdir()) if i.name.endswith("00")]
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        # executor.map(proposed_fusion, videos, range(len(videos)))
        executor.map(proposed_fusion, videos, range(len(videos)))


if __name__ == "__main__":
    main()
