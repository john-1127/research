time python chempropIRZenodo/chempropIR/predict.py \
  --gpu 0 \
  --qnn \
  --features_generator morgan \
  --test_path ./nist/predict_smiles.csv \
  --checkpoint_path ./output/model/qh2_2100_layer3/fold_0/model_0/model.pt \
  --preds_path ./test1234.csv
