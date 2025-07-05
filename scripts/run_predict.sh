time python chempropIRZenodo/chempropIR/predict.py \
  --gpu 0 \
  --features_generator morgan \
  --test_path ./output/model/qnn_first/fold_0/test_smiles.csv \
  --checkpoint_path ./output/model/ffnn_pretrained/fold_0/model_0/model.pt \
  --preds_path ./output/model/ffnn_pretrained/fold_0/ffnn_pretrained.csv

# --qnn
