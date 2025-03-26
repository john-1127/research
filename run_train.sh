#!/bin/bash

time python chempropIRZenodo/chempropIR/train.py \
  --gpu 0 \
  --data_path ./data/research_fingerprint_data/train_full.csv \
  --separate_val_path ./data/research_fingerprint_data/val_full.csv \
  --separate_test_path ./data/research_fingerprint_data/test_full.csv \
  --features_only \
  --qnn \
  --features_generator morgan \
  --dataset_type spectra \
  --save_dir ./output/model/morgan_hybrid_fingerprint_default_lr \
  --config_path ./recommended_config.json

# if we don't want to seperate data auto, assign path
# --separate_val_path \
# --separate_test_path \

# --frzn_mpn_checkpoint model/test/model_0.pt
