#!/bin/bash

time python chempropIRZenodo/chempropIR/train.py \
  --gpu 0 \
  --data_path ./data/research_data/train_full.csv \
  --separate_val_path ./data/research_data/val_full.csv \
  --separate_test_path ./data/research_data/test_full.csv \
  --features_only \
  --qnn \
  --qnn_layer 2 \
  --features_generator morgan \
  --dataset_type spectra \
  --save_dir ./output/model/qh2_2048_layer1_1000 \
  --config_path ./recommended_config.json

# if we don't want to seperate data auto, assign path
# --separate_val_path \
# --separate_test_path \

# --frzn_mpn_checkpoint model/test/model_0.pt
