#!/bin/bash

DATASET="$HOME/dataset/"
VOCABULARY="scan_net"
CONFIDENCE_THRESH=0.3

if [ "$1" = "0" ]; then
    VIDEOS=(scene0011_00 scene0084_00 scene0164_00 scene0231_00 scene0314_00 scene0355_00 scene0426_00 scene0490_00
            scene0558_00 scene0595_00 scene0633_00 scene0660_00 scene0690_00 scene0015_00 scene0086_00 scene0169_00
            scene0246_00 scene0316_00)
    for VIDEO in "${VIDEOS[@]}"; do
        echo "processing video: $VIDEO"
        CUDA_VISIBLE_DEVICES=$1 python demo_custom.py \
            --config-file "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml" \
            --dataset "$DATASET" \
            --video "$VIDEO" \
            --output_folder "detic_output" \
            --vocabulary "$VOCABULARY" \
            --confidence-threshold $CONFIDENCE_THRESH \
            --opts MODEL.WEIGHTS "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    done
elif [ "$1" = "1" ]; then
    VIDEOS=(scene0356_00 scene0427_00 scene0494_00 scene0559_00 scene0598_00 scene0643_00 scene0663_00 scene0693_00
            scene0019_00 scene0088_00 scene0187_00 scene0249_00 scene0328_00 scene0357_00 scene0430_00 scene0496_00
            scene0565_00 scene0599_00)
    for VIDEO in "${VIDEOS[@]}"; do
        echo "processing video: $VIDEO"
        CUDA_VISIBLE_DEVICES=$1 python demo_custom.py \
            --config-file "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml" \
            --dataset "$DATASET" \
            --video "$VIDEO" \
            --output_folder "detic_output" \
            --vocabulary "$VOCABULARY" \
            --confidence-threshold $CONFIDENCE_THRESH \
            --opts MODEL.WEIGHTS "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    done
elif [ "$1" = "2" ]; then
    VIDEOS=(scene0644_00 scene0664_00 scene0695_00 scene0025_00 scene0095_00 scene0193_00 scene0251_00 scene0329_00
            scene0377_00 scene0432_00 scene0500_00 scene0568_00 scene0606_00 scene0645_00 scene0665_00 scene0696_00
            scene0030_00 scene0100_00)
    for VIDEO in "${VIDEOS[@]}"; do
        echo "processing video: $VIDEO"
        CUDA_VISIBLE_DEVICES=$1 python demo_custom.py \
            --config-file "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml" \
            --dataset "$DATASET" \
            --video "$VIDEO" \
            --output_folder "detic_output" \
            --vocabulary "$VOCABULARY" \
            --confidence-threshold $CONFIDENCE_THRESH \
            --opts MODEL.WEIGHTS "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    done
elif [ "$1" = "3" ]; then
    VIDEOS=(scene0196_00 scene0256_00 scene0334_00 scene0378_00 scene0435_00 scene0518_00 scene0574_00 scene0607_00
            scene0647_00 scene0670_00 scene0697_00 scene0046_00 scene0131_00 scene0203_00 scene0257_00 scene0338_00
            scene0382_00 scene0441_00)
    for VIDEO in "${VIDEOS[@]}"; do
        echo "processing video: $VIDEO"
        CUDA_VISIBLE_DEVICES=$1 python demo_custom.py \
            --config-file "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml" \
            --dataset "$DATASET" \
            --video "$VIDEO" \
            --output_folder "detic_output" \
            --vocabulary "$VOCABULARY" \
            --confidence-threshold $CONFIDENCE_THRESH \
            --opts MODEL.WEIGHTS "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    done
elif [ "$1" = "4" ]; then
    VIDEOS=(scene0527_00 scene0575_00 scene0608_00 scene0648_00 scene0671_00 scene0699_00 scene0050_00 scene0139_00
            scene0207_00 scene0277_00 scene0342_00 scene0389_00 scene0458_00 scene0535_00 scene0578_00 scene0609_00
            scene0651_00 scene0678_00)
    for VIDEO in "${VIDEOS[@]}"; do
        echo "processing video: $VIDEO"
        CUDA_VISIBLE_DEVICES=$1 python demo_custom.py \
            --config-file "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml" \
            --dataset "$DATASET" \
            --video "$VIDEO" \
            --output_folder "detic_output" \
            --vocabulary "$VOCABULARY" \
            --confidence-threshold $CONFIDENCE_THRESH \
            --opts MODEL.WEIGHTS "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    done
elif [ "$1" = "5" ]; then
    VIDEOS=(scene0700_00 scene0063_00 scene0144_00 scene0208_00 scene0278_00 scene0343_00 scene0406_00 scene0461_00
            scene0549_00 scene0580_00 scene0616_00 scene0652_00 scene0684_00 scene0701_00 scene0064_00 scene0146_00
            scene0217_00 scene0300_00)
    for VIDEO in "${VIDEOS[@]}"; do
        echo "processing video: $VIDEO"
        CUDA_VISIBLE_DEVICES=$1 python demo_custom.py \
            --config-file "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml" \
            --dataset "$DATASET" \
            --video "$VIDEO" \
            --output_folder "detic_output" \
            --vocabulary "$VOCABULARY" \
            --confidence-threshold $CONFIDENCE_THRESH \
            --opts MODEL.WEIGHTS "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    done
elif [ "$1" = "6" ]; then
    VIDEOS=(scene0351_00 scene0412_00 scene0462_00 scene0550_00 scene0583_00 scene0618_00 scene0653_00 scene0685_00
            scene0702_00 scene0077_00 scene0149_00 scene0221_00 scene0304_00 scene0353_00 scene0414_00 scene0474_00
            scene0552_00 scene0591_00)
    for VIDEO in "${VIDEOS[@]}"; do
        echo "processing video: $VIDEO"
        CUDA_VISIBLE_DEVICES=$1 python demo_custom.py \
            --config-file "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml" \
            --dataset "$DATASET" \
            --video "$VIDEO" \
            --output_folder "detic_output" \
            --vocabulary "$VOCABULARY" \
            --confidence-threshold $CONFIDENCE_THRESH \
            --opts MODEL.WEIGHTS "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    done
elif [ "$1" = "7" ]; then
    VIDEOS=(scene0621_00 scene0655_00 scene0686_00 scene0704_00 scene0081_00 scene0153_00 scene0222_00 scene0307_00
            scene0354_00 scene0423_00 scene0488_00 scene0553_00 scene0593_00 scene0629_00 scene0658_00 scene0689_00)
    for VIDEO in "${VIDEOS[@]}"; do
        echo "processing video: $VIDEO"
        CUDA_VISIBLE_DEVICES=$1 python demo_custom.py \
            --config-file "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml" \
            --dataset "$DATASET" \
            --video "$VIDEO" \
            --output_folder "detic_output" \
            --vocabulary "$VOCABULARY" \
            --confidence-threshold $CONFIDENCE_THRESH \
            --opts MODEL.WEIGHTS "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    done
fi
