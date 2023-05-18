echo $$
CUDA_DEVICE=$1
DMOVE=$2
DLG=$3
# SEED=$2
# INT=$3

echo $$ $CUDA_DEVICE $DMOVE $DLG



FOLDER="models/incremental_pretrained_3"
# FOLDER="models/incremental"
# FOLDER="models/incremental_grad_clip_10"
mkdir -p $FOLDER
# CUDA_DEVICE=0
for SEED in 0 1 2 3 4 5 6 7 8 9 ; do #1 2 3 4 5; do #0; do # 
    for MODEL in LSTM; do # LSTM; do # Transformer; do # 
        for EXP in 3; do # 2 3; do
            # for DMOVE in "No" "Yes"; do
            #     for DLG in "No" "Yes"; do
                    for INT in 0 1 2 4 3 5 6 7; do
                        FILE_NAME="plan_exp${EXP}_${MODEL}_dlg_${DLG}_move_${DMOVE}_int${INT}_seed_${SEED}"
                        COMM="plan_predictor.py"
                        COMM=$COMM" --seed=${SEED}"
                        COMM=$COMM" --device=${CUDA_DEVICE}"
                        COMM=$COMM" --seed=${SEED}"
                        COMM=$COMM" --use_dialogue_moves=${DMOVE}"
                        COMM=$COMM" --use_dialogue=${DLG}"
                        COMM=$COMM" --experiment=${EXP}"
                        COMM=$COMM" --seq_model=${MODEL}"
                        COMM=$COMM" --pov=First"
                        COMM=$COMM" --plan=Yes"
                        COMM=$COMM" --intermediate=${INT}"
                        if [ $INT -gt 0 ]; then
                            COMM=$COMM" --model_path=${FOLDER}/plan_exp${EXP}_${MODEL}_dlg_${DLG}_move_${DMOVE}_int0_seed_${SEED}.torch"
                        fi
                        COMM=$COMM" --save_path=${FOLDER}/${FILE_NAME}.torch"
                        echo $(date +%F\ %T)" python3 ${COMM} > ${FOLDER}/${FILE_NAME}.log"
                        python3 $COMM > ${FOLDER}/${FILE_NAME}.log
                    done
            #     done
            # done
        done
    done
done



echo 'Done!'


















