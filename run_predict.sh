python predict.py \
  --gpu 0 \
  --test_path trained_ir_model/experiment_model/test_smiles.csv \
  --features_path trained_ir_model/experiment_model/test_features.csv \
  --checkpoint_dir trained_ir_model/experiment_model/model_files \
  --preds_path ./output/ir_preds.csv
