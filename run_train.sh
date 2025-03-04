#!/bin/bash

time python chempropIRZenodo/chempropIR/train.py \
  --gpu 0 \
  --data_path chempropIRZenodo/trained_ir_model/computed_model/test_full.csv \
  --features_only \
  --features_generator morgan \
  --dataset_type spectra \
  --test \
  --qnn \
  --save_dir ./output/test \
  --config_path ./recommended_config.json

# if we don't want to seperate data auto, assign path
# --separate_val_path \
# --separate_test_path \

# --frzn_mpn_checkpoint model/test/model_0.pt
