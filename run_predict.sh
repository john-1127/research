time python chempropIRZenodo/chempropIR/predict.py \
  --gpu 0 \
  --qnn \
  --features_generator morgan \
  --test_path ./data/research_data/test_smiles.csv \
  --checkpoint_path output/model/morgan_hybrid_fingerprint/fold_0/model_0/model.pt \
  --preds_path output/model/morgan_hybrid_fingerprint/fold_0/morgan_hybrid_fingerprint.csv
