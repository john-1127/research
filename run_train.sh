#!/bin/bash

python chempropIRZenodo/chempropIR/train.py \
  --gpu 0 \
  --data_path chempropIRZenodo/trained_ir_model/computed_model/test_full.csv \
  --dataset_type spectra \
  --save_dir ./output \
  --config_path chempropIRZenodo/trained_ir_model/recommended_config.json
