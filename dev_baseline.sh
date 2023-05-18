# python3 baselines_with_dialogue_moves.py --pov=First --use_dialogue=Yes --plans=Yes --seq_model=LSTM --experiment=6 --seed=Fixed --save_path=text.torch

# FOLDER="models/gt_dialogue_moves_good_flip"
# mkdir -p $FOLDER
# SEED=0
# CUDA_DEVICE=$1
# for MODEL in LSTM Transformer; do # Transformer; do
#     for POV in None First; do
#         for DLG in No Yes; do
#             for DLGM in No Yes; do
#                 for EXP in 6 7 8; do # 2 3; do
#                     FILE_NAME="gt_dialogue_moves_${MODEL}_dlgMove_${DLGM}_dlg_${DLG}_pov_${POV}_exp${EXP}_seed_${SEED}"
#                     COMM="baselines_with_dialogue_moves.py --seed=${SEED} --device=${CUDA_DEVICE}"
#                     COMM=$COMM" --use_dialogue=${DLG} --use_dialogue_moves=${DLGM}"
#                     COMM=$COMM" --experiment=${EXP} --seq_model=${MODEL} --pov=${POV}"
#                     COMM=$COMM" --plan=Yes --save_path=${FOLDER}/${FILE_NAME}.torch"
#                     echo $(date +%F\ %T) $COMM" > ${FOLDER}/${FILE_NAME}.log"
#                     # python3 $COMM > ${FOLDER}/${FILE_NAME}.log
#                 done
#             done
#         done
#     done
# done
# echo "Done!"

nohup ./baselines_with_dialogue_moves.sh 0 0 &
nohup ./baselines_with_dialogue_moves.sh 1 1 &
nohup ./baselines_with_dialogue_moves.sh 0 2 &
nohup ./baselines_with_dialogue_moves.sh 1 3 &
nohup ./baselines_with_dialogue_moves.sh 0 4 &
nohup ./baselines_with_dialogue_moves.sh 1 5 &
