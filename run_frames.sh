#!/bin/bash

if [ "$1" = "1" ]; then
    VIDEOS=({0030..0034})
    for VIDEO in ${VIDEOS[@]}; do
        echo "processing video: $VIDEO"
        CUDA_VISIBLE_DEVICES=$1 python demo_custom.py \
            --config-file "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml" \
            --dataset "$HOME/dataset/ycb_video" \
            --video "$VIDEO" \
            --output_folder "detic_output" \
            --vocabulary "ycb_video" \
            --confidence-threshold 0.3 \
            --opts MODEL.WEIGHTS "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    done
elif [ "$1" = "2" ]; then
    VIDEOS=({0035..0039})
    for VIDEO in ${VIDEOS[@]}; do
        echo "processing video: $VIDEO"
        CUDA_VISIBLE_DEVICES=$1 python demo_custom.py \
            --config-file "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml" \
            --dataset "$HOME/dataset/ycb_video" \
            --video "$VIDEO" \
            --output_folder "detic_output" \
            --vocabulary "ycb_video" \
            --confidence-threshold 0.3 \
            --opts MODEL.WEIGHTS "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    done
elif [ "$1" = "3" ]; then
    VIDEOS=({0040..0044})
    for VIDEO in ${VIDEOS[@]}; do
        echo "processing video: $VIDEO"
        CUDA_VISIBLE_DEVICES=$1 python demo_custom.py \
            --config-file "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml" \
            --dataset "$HOME/dataset/ycb_video" \
            --video "$VIDEO" \
            --output_folder "detic_output" \
            --vocabulary "ycb_video" \
            --confidence-threshold 0.3 \
            --opts MODEL.WEIGHTS "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    done
elif [ "$1" = "5" ]; then
    VIDEOS=({0045..0049})
    for VIDEO in ${VIDEOS[@]}; do
        echo "processing video: $VIDEO"
        CUDA_VISIBLE_DEVICES=$1 python demo_custom.py \
            --config-file "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml" \
            --dataset "$HOME/dataset/ycb_video" \
            --video "$VIDEO" \
            --output_folder "detic_output" \
            --vocabulary "ycb_video" \
            --confidence-threshold 0.3 \
            --opts MODEL.WEIGHTS "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    done
elif [ "$1" = "6" ]; then
    VIDEOS=({0050..0054})
    for VIDEO in ${VIDEOS[@]}; do
        echo "processing video: $VIDEO"
        CUDA_VISIBLE_DEVICES=$1 python demo_custom.py \
            --config-file "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml" \
            --dataset "$HOME/dataset/ycb_video" \
            --video "$VIDEO" \
            --output_folder "detic_output" \
            --vocabulary "ycb_video" \
            --confidence-threshold 0.3 \
            --opts MODEL.WEIGHTS "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    done
elif [ "$1" = "7" ]; then
    VIDEOS=({0055..0059})
    for VIDEO in ${VIDEOS[@]}; do
        echo "processing video: $VIDEO"
        CUDA_VISIBLE_DEVICES=$1 python demo_custom.py \
            --config-file "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml" \
            --dataset "$HOME/dataset/ycb_video" \
            --video "$VIDEO" \
            --output_folder "detic_output" \
            --vocabulary "ycb_video" \
            --confidence-threshold 0.3 \
            --opts MODEL.WEIGHTS "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    done
fi

