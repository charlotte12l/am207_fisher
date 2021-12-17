#!/bin/bash
REPO_ROOT="/Users/charlotte/Desktop/study/am207/group/am207_fisher"
RESULT_FOLDER=$REPO_ROOT/proj_output
DATAPATH=$REPO_ROOT/data/ml_privacy_csf18

### IWPC EXPERIMENTS ###

DATASET="iwpc"
MODEL="least_squares"

# For L2 and sigma inversion plots:
for L2 in "1e-5" "1e-3" "1e-1" "1"
do
  FIL_RESULTS="${RESULT_FOLDER}/${DATASET}_${MODEL}_fil_l2_${L2}"
  python $REPO_ROOT/reweighted.py \
    --data_folder $DATAPATH \
    --dataset $DATASET \
    --model $MODEL \
    --pca_dims 0 \
    --no_norm \
    --l2 $L2 \
    --attribute 11 13 \
    --results_file $FIL_RESULTS

  INVERSION_RESULTS="${RESULT_FOLDER}/${DATASET}_${MODEL}_inversion_l2_${L2}.json"
  python $REPO_ROOT/model_inversion.py \
    --data_folder $DATAPATH \
    --inverter all \
    --dataset $DATASET \
    --model $MODEL \
    --l2 $L2 \
    --results_file $INVERSION_RESULTS
  
  for INVERTER in 'fredrikson14' 'whitebox'
  do
    INVERSION_RESULTS="${RESULT_FOLDER}/${DATASET}_${MODEL}_${INVERTER}_private_inversion_l2_${L2}.json"
    python $REPO_ROOT/private_model_inversion.py \
      --data_folder $DATAPATH \
      --dataset $DATASET \
      --trials 100 \
      --noise_scales 1e-5 2e-5 5e-5 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2 2e-2 5e-2 1e-1 2e-1 5e-1 1 \
      --inverter $INVERTER \
      --model $MODEL \
      --l2 $L2 \
      --results_file $INVERSION_RESULTS
  done
done

# For IRFIL inversion plots:
L2=1e-2
IRFIL_RESULTS="${RESULT_FOLDER}/${DATASET}_${MODEL}_irfil"
python $REPO_ROOT/reweighted.py \
  --data_folder $DATAPATH \
  --dataset $DATASET \
  --model $MODEL \
  --pca_dims 0 \
  --iters 10 \
  --no_norm \
  --l2 $L2 \
  --attribute 11 13 \
  --results_file $IRFIL_RESULTS

for INVERTER in 'fredrikson14' 'whitebox'
do
  INVERSION_RESULTS="${RESULT_FOLDER}/${DATASET}_${MODEL}_${INVERTER}_private_inversion_irfil.json"
  python $REPO_ROOT/private_model_inversion.py \
    --data_folder $DATAPATH \
    --dataset $DATASET \
    --trials 100 \
    --noise_scales 1e-4 1e-3 1e-2 \
    --inverter $INVERTER \
    --model $MODEL \
    --l2 $L2 \
    --weights_file ${IRFIL_RESULTS}.pth \
    --results_file $INVERSION_RESULTS
done