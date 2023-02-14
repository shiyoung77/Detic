#!/bin/bash

VIDEOS=({0001..0030})
for VIDEO in ${VIDEOS[@]}; do
    echo "processing video: $VIDEO"
    python demo_custom.py \
        --config-file "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml" \
        --dataset "$HOME/dataset/CoRL_real" \
        --video "$VIDEO" \
        --output_folder "detic_output" \
        --vocabulary "icra23" \
        --confidence-threshold 0.3 \
        --opts MODEL.WEIGHTS "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
done
