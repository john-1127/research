#!/bin/bash

time python ./chempropIRZenodo/chempropIR/hyperparameter_optimization.py \
  --gpu 0 \
  --init_lr 5e-5 \
  --final_lr 5e-5 \
  --data_path ./data/research_data/train_full.csv \
  --separate_val_path ./data/research_data/val_full.csv \
  --separate_test_path ./data/research_data/test_full.csv \
  --features_only \
  --qnn \
  --features_generator morgan \
  --dataset_type spectra \
  --save_dir ./output/hyperparameters \
  --config_save_path ./output/hyperparameters/best/best.json \
  --config_path ./recommended_config.json

# if we don't want to seperate data auto, assign path
# --separate_val_path \
# --separate_test_path \

# --frzn_mpn_checkpoint model/test/model_0.pt
