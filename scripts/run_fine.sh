#!/bin/bash

time python chempropIRZenodo/chempropIR/train.py \
  --gpu 0 \
  --data_path ./output/model/qnn_fine_tunning/fold_0/train_full.csv \
  --separate_val_path ./output/model/qnn_fine_tunning/fold_0/val_full.csv \
  --separate_test_path ./output/model/qnn_fine_tunning/fold_0/test_full.csv \
  --features_only \
  --qnn \
  --qnn_layer 2 \
  --features_generator morgan \
  --dataset_type spectra \
  --save_dir ./output/model/test_fine_tunning \
  --checkpoint_path ./output/model/qh2_2100_layer3/fold_0/model_0/model.pt \
  --config_path ./recommended_config.json

# if we don't want to seperate data auto, assign path
# --separate_val_path \
# --separate_test_path \

# --frzn_mpn_checkpoint model/test/model_0.pt
