#!/bin/bash

time python chempropIRZenodo/chempropIR/train.py \
  --gpu 0 \
  --data_path chempropIRZenodo/trained_ir_model/computed_model/test_full.csv \
  --dataset_type spectra \
  --save_dir ./output/model \
  --config_path ./recommended_config.json \
  --frzn_mpn_checkpoint model/test/model_0.pt

# if we don't want to seperate data auto, assign path
# --separate_val_path \
# --separate_test_path \
