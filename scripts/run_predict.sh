time python chempropIRZenodo/chempropIR/predict.py \
  --gpu 0 \
  --features_generator morgan \
  --test_path ./data/research_data/train_smiles.csv \
  --checkpoint_path ./output/model/classical_2100_layer3/fold_0/model_0/model.pt \
  --preds_path ./output/model/classical_2100_layer3/fold_0/classical_2100_layer3_train.csv
