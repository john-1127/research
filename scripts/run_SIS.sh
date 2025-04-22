X=$1
GROUP=$2
MODEL=$3
FFNN=$4
HQNN=$5

if [[ "$MODEL" == "FFNN" ]]; then
  python chempropIRZenodo/chempropIR/predict.py \
    --gpu 0 \
    --features_generator morgan \
    --test_path ./data/research_data/test_smiles.csv \
    --checkpoint_path output/model/$X/fold_0/model_0/model.pt \
    --preds_path output/model/$X/fold_0/$X.csv

elif [[ "$MODEL" == "HQNN" ]]; then
  python chempropIRZenodo/chempropIR/predict.py \
    --gpu 0 \
    --qnn \
    --features_generator morgan \
    --test_path ./data/research_data/test_smiles.csv \
    --checkpoint_path output/model/$X/fold_0/model_0/model.pt \
    --preds_path output/model/$X/fold_0/$X.csv

elif [[ "$MODEL" == "Ensemble" ]]; then
  python ./scripts/ensemble.py $FFNN $HQNN
fi

python ./chempropIRZenodo/chempropIR/scripts/SIS_spectra_similarity.py ./output/model/$X/fold_0/$X.csv ./data/research_data/test_full.csv

python ./output/sis/sis_summary.py $GROUP $MODEL ./output/sis/sis_summary.csv

mv ./output/sis/similarity.txt ./output/sis/$X.txt
