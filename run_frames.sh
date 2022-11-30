#!/bin/bash

python demo_custom.py \
  --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
  --dataset "/home/lsy/dataset/CoRL_real" \
  --video "0001" \
  --output_folder "detic_output" \
  --vocabulary lvis \
  --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
