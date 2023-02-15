#!/bin/bash

VIDEOS=({0000..0001})
for VIDEO in ${VIDEOS[@]}; do
    echo "processing video: $VIDEO"
    python demo_custom.py \
        --config-file "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml" \
        --dataset "$HOME/dataset/ycb_video" \
        --video "$VIDEO" \
        --output_folder "detic_output" \
        --vocabulary "ycb_video" \
        --confidence-threshold 0.3 \
        --opts MODEL.WEIGHTS "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
done
