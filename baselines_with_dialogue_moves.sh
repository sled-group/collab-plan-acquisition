# python3 baselines_with_dialogue_moves.py --pov=First --use_dialogue=Yes --plans=Yes --seq_model=LSTM --experiment=6 --seed=Fixed --save_path=text.torch

# FOLDER="models/gt_dialogue_moves2"
# mkdir -p $FOLDER
# CUDA_DEVICE=$1
# SEED=$2
# for MODEL in LSTM; do # LSTM Transformer; do # Transformer; do # 
#     for DLGM in Yes; do # No Yes; do # No; do # 
#         for DLG in No Yes; do
#             for POV in First None; do
#                 for EXP in 6 7 8; do # 2 3; do
#                     FILE_NAME="gt_dialogue_moves_${MODEL}_dlgMove_${DLGM}_dlg_${DLG}_pov_${POV}_exp${EXP}_seed_${SEED}"
#                     COMM="baselines_with_dialogue_moves.py"
#                     COMM=$COMM" --device=${CUDA_DEVICE}"
#                     COMM=$COMM" --seed=${SEED}"
#                     COMM=$COMM" --use_dialogue_moves=${DLGM}"
#                     COMM=$COMM" --use_dialogue=${DLG}"
#                     COMM=$COMM" --pov=${POV}"
#                     COMM=$COMM" --experiment=${EXP}"
#                     COMM=$COMM" --seq_model=${MODEL}"
#                     COMM=$COMM" --plan=Yes"
#                     COMM=$COMM" --save_path=${FOLDER}/${FILE_NAME}.torch"
#                     echo $(date +%F\ %T) $COMM" > ${FOLDER}/${FILE_NAME}.log"
#                     python3 $COMM > ${FOLDER}/${FILE_NAME}.log
#                 done
#             done
#         done
#     done
# done
# echo "Done!"




# FOLDER="models/gt_dialogue_moves_bootstrap"
# mkdir -p $FOLDER
# CUDA_DEVICE=$1
# SEED=$2

# for MODEL in Transformer; do # LSTM; do # LSTM Transformer; do # 
#     for DLGM in No; do # Yes; do # No Yes; do # 
#         for EXP in 6 7 8; do # 2 3; do

#             DLG="Yes"
#             POV="None"

#             FILE_NAME="gt_dialogue_moves_${MODEL}_dlgMove_${DLGM}_dlg_${DLG}_pov_${POV}_exp${EXP}_seed_${SEED}"
#             COMM="baselines_with_dialogue_moves.py"
#             COMM=$COMM" --device=${CUDA_DEVICE}"
#             COMM=$COMM" --seed=${SEED}"
#             COMM=$COMM" --use_dialogue_moves=${DLGM}"
#             COMM=$COMM" --use_dialogue=${DLG}"
#             COMM=$COMM" --pov=${POV}"
#             COMM=$COMM" --experiment=${EXP}"
#             COMM=$COMM" --seq_model=${MODEL}"
#             COMM=$COMM" --plan=Yes"
#             COMM=$COMM" --save_path=${FOLDER}/${FILE_NAME}.torch"
#             echo $(date +%F\ %T) $COMM" > ${FOLDER}/${FILE_NAME}.log"
#             python3 $COMM > ${FOLDER}/${FILE_NAME}.log

#             DLG="No"
#             POV="First"

#             FILE_NAME="gt_dialogue_moves_${MODEL}_dlgMove_${DLGM}_dlg_${DLG}_pov_${POV}_exp${EXP}_seed_${SEED}"
#             COMM="baselines_with_dialogue_moves.py"
#             COMM=$COMM" --device=${CUDA_DEVICE}"
#             COMM=$COMM" --seed=${SEED}"
#             COMM=$COMM" --use_dialogue_moves=${DLGM}"
#             COMM=$COMM" --use_dialogue=${DLG}"
#             COMM=$COMM" --pov=${POV}"
#             COMM=$COMM" --experiment=${EXP}"
#             COMM=$COMM" --seq_model=${MODEL}"
#             COMM=$COMM" --plan=Yes"
#             COMM=$COMM" --save_path=${FOLDER}/${FILE_NAME}.torch"
#             echo $(date +%F\ %T) $COMM" > ${FOLDER}/${FILE_NAME}.log"
#             python3 $COMM > ${FOLDER}/${FILE_NAME}.log

#             DLG="Yes"
#             POV="First"

#             FILE_NAME="gt_dialogue_moves_${MODEL}_dlgMove_${DLGM}_dlg_${DLG}_pov_${POV}_exp${EXP}_seed_${SEED}_DlgFirst"
#             COMM="baselines_with_dialogue_moves.py"
#             COMM=$COMM" --device=${CUDA_DEVICE}"
#             COMM=$COMM" --seed=${SEED}"
#             COMM=$COMM" --use_dialogue_moves=${DLGM}"
#             COMM=$COMM" --use_dialogue=${DLG}"
#             COMM=$COMM" --pov=${POV}"
#             COMM=$COMM" --experiment=${EXP}"
#             COMM=$COMM" --seq_model=${MODEL}"
#             COMM=$COMM" --plan=Yes"
#             COMM=$COMM" --model_path=${FOLDER}/gt_dialogue_moves_${MODEL}_dlgMove_${DLGM}_dlg_Yes_pov_None_exp${EXP}_seed_${SEED}.torch"
#             COMM=$COMM" --save_path=${FOLDER}/${FILE_NAME}.torch"
#             echo $(date +%F\ %T) $COMM" > ${FOLDER}/${FILE_NAME}.log"
#             python3 $COMM > ${FOLDER}/${FILE_NAME}.log

#             DLG="Yes"
#             POV="First"

#             FILE_NAME="gt_dialogue_moves_${MODEL}_dlgMove_${DLGM}_dlg_${DLG}_pov_${POV}_exp${EXP}_seed_${SEED}_VidFirst"
#             COMM="baselines_with_dialogue_moves.py"
#             COMM=$COMM" --device=${CUDA_DEVICE}"
#             COMM=$COMM" --seed=${SEED}"
#             COMM=$COMM" --use_dialogue_moves=${DLGM}"
#             COMM=$COMM" --use_dialogue=${DLG}"
#             COMM=$COMM" --pov=${POV}"
#             COMM=$COMM" --experiment=${EXP}"
#             COMM=$COMM" --seq_model=${MODEL}"
#             COMM=$COMM" --plan=Yes"
#             COMM=$COMM" --model_path=${FOLDER}/gt_dialogue_moves_${MODEL}_dlgMove_${DLGM}_dlg_No_pov_First_exp${EXP}_seed_${SEED}.torch"
#             COMM=$COMM" --save_path=${FOLDER}/${FILE_NAME}.torch"
#             echo $(date +%F\ %T) $COMM" > ${FOLDER}/${FILE_NAME}.log"
#             python3 $COMM > ${FOLDER}/${FILE_NAME}.log
#         done
#     done
# done


# for MODEL in Transformer; do # LSTM; do # LSTM Transformer; do # 
#     for DLGM in Yes; do # No; do # Yes; do # No 
#         for EXP in 6 7 8; do # 2 3; do

#             DLG="No"
#             POV="None"

#             FILE_NAME="gt_dialogue_moves_${MODEL}_dlgMove_${DLGM}_dlg_${DLG}_pov_${POV}_exp${EXP}_seed_${SEED}"
#             COMM="baselines_with_dialogue_moves.py"
#             COMM=$COMM" --device=${CUDA_DEVICE}"
#             COMM=$COMM" --seed=${SEED}"
#             COMM=$COMM" --use_dialogue_moves=${DLGM}"
#             COMM=$COMM" --use_dialogue=${DLG}"
#             COMM=$COMM" --pov=${POV}"
#             COMM=$COMM" --experiment=${EXP}"
#             COMM=$COMM" --seq_model=${MODEL}"
#             COMM=$COMM" --plan=Yes"
#             COMM=$COMM" --save_path=${FOLDER}/${FILE_NAME}.torch"
#             echo $(date +%F\ %T) $COMM" > ${FOLDER}/${FILE_NAME}.log"
#             python3 $COMM > ${FOLDER}/${FILE_NAME}.log

#             DLG="Yes"
#             POV="None"

#             FILE_NAME="gt_dialogue_moves_${MODEL}_dlgMove_${DLGM}_dlg_${DLG}_pov_${POV}_exp${EXP}_seed_${SEED}"
#             COMM="baselines_with_dialogue_moves.py"
#             COMM=$COMM" --device=${CUDA_DEVICE}"
#             COMM=$COMM" --seed=${SEED}"
#             COMM=$COMM" --use_dialogue_moves=${DLGM}"
#             COMM=$COMM" --use_dialogue=${DLG}"
#             COMM=$COMM" --pov=${POV}"
#             COMM=$COMM" --experiment=${EXP}"
#             COMM=$COMM" --seq_model=${MODEL}"
#             COMM=$COMM" --plan=Yes"
#             COMM=$COMM" --model_path=${FOLDER}/gt_dialogue_moves_${MODEL}_dlgMove_No_dlg_${DLG}_pov_${POV}_exp${EXP}_seed_${SEED}.torch"
#             COMM=$COMM" --save_path=${FOLDER}/${FILE_NAME}.torch"
#             echo $(date +%F\ %T) $COMM" > ${FOLDER}/${FILE_NAME}.log"
#             python3 $COMM > ${FOLDER}/${FILE_NAME}.log

#             DLG="No"
#             POV="First"

#             FILE_NAME="gt_dialogue_moves_${MODEL}_dlgMove_${DLGM}_dlg_${DLG}_pov_${POV}_exp${EXP}_seed_${SEED}"
#             COMM="baselines_with_dialogue_moves.py"
#             COMM=$COMM" --device=${CUDA_DEVICE}"
#             COMM=$COMM" --seed=${SEED}"
#             COMM=$COMM" --use_dialogue_moves=${DLGM}"
#             COMM=$COMM" --use_dialogue=${DLG}"
#             COMM=$COMM" --pov=${POV}"
#             COMM=$COMM" --experiment=${EXP}"
#             COMM=$COMM" --seq_model=${MODEL}"
#             COMM=$COMM" --plan=Yes"
#             COMM=$COMM" --model_path=${FOLDER}/gt_dialogue_moves_${MODEL}_dlgMove_No_dlg_${DLG}_pov_${POV}_exp${EXP}_seed_${SEED}.torch"
#             COMM=$COMM" --save_path=${FOLDER}/${FILE_NAME}.torch"
#             echo $(date +%F\ %T) $COMM" > ${FOLDER}/${FILE_NAME}.log"
#             python3 $COMM > ${FOLDER}/${FILE_NAME}.log

#             DLG="Yes"
#             POV="First"

#             FILE_NAME="gt_dialogue_moves_${MODEL}_dlgMove_${DLGM}_dlg_${DLG}_pov_${POV}_exp${EXP}_seed_${SEED}_DlgFirst"
#             COMM="baselines_with_dialogue_moves.py"
#             COMM=$COMM" --device=${CUDA_DEVICE}"
#             COMM=$COMM" --seed=${SEED}"
#             COMM=$COMM" --use_dialogue_moves=${DLGM}"
#             COMM=$COMM" --use_dialogue=${DLG}"
#             COMM=$COMM" --pov=${POV}"
#             COMM=$COMM" --experiment=${EXP}"
#             COMM=$COMM" --seq_model=${MODEL}"
#             COMM=$COMM" --plan=Yes"
#             COMM=$COMM" --model_path=${FOLDER}/gt_dialogue_moves_${MODEL}_dlgMove_No_dlg_${DLG}_pov_${POV}_exp${EXP}_seed_${SEED}_DlgFirst.torch"
#             COMM=$COMM" --save_path=${FOLDER}/${FILE_NAME}.torch"
#             echo $(date +%F\ %T) $COMM" > ${FOLDER}/${FILE_NAME}.log"
#             python3 $COMM > ${FOLDER}/${FILE_NAME}.log

#             DLG="Yes"
#             POV="First"

#             FILE_NAME="gt_dialogue_moves_${MODEL}_dlgMove_${DLGM}_dlg_${DLG}_pov_${POV}_exp${EXP}_seed_${SEED}_VidFirst"
#             COMM="baselines_with_dialogue_moves.py"
#             COMM=$COMM" --device=${CUDA_DEVICE}"
#             COMM=$COMM" --seed=${SEED}"
#             COMM=$COMM" --use_dialogue_moves=${DLGM}"
#             COMM=$COMM" --use_dialogue=${DLG}"
#             COMM=$COMM" --pov=${POV}"
#             COMM=$COMM" --experiment=${EXP}"
#             COMM=$COMM" --seq_model=${MODEL}"
#             COMM=$COMM" --plan=Yes"
#             COMM=$COMM" --model_path=${FOLDER}/gt_dialogue_moves_${MODEL}_dlgMove_No_dlg_${DLG}_pov_${POV}_exp${EXP}_seed_${SEED}_VidFirst.torch"
#             COMM=$COMM" --save_path=${FOLDER}/${FILE_NAME}.torch"
#             echo $(date +%F\ %T) $COMM" > ${FOLDER}/${FILE_NAME}.log"
#             python3 $COMM > ${FOLDER}/${FILE_NAME}.log

#         done
#     done
# done


# echo "Done!"






FOLDER="models/gt_dialogue_moves_bootstrap_DlgMoveFirst"
mkdir -p $FOLDER
CUDA_DEVICE=$1
SEED=$2

for MODEL in Transformer; do # LSTM; do # LSTM Transformer; do # 
    for DLGM in Yes; do # No; do # Yes; do # No 
        for EXP in 6 7 8; do # 2 3; do

            DLG="No"
            POV="None"

            FILE_NAME="gt_dialogue_moves_${MODEL}_dlgMove_${DLGM}_dlg_${DLG}_pov_${POV}_exp${EXP}_seed_${SEED}"
            COMM="baselines_with_dialogue_moves.py"
            COMM=$COMM" --device=${CUDA_DEVICE}"
            COMM=$COMM" --seed=${SEED}"
            COMM=$COMM" --use_dialogue_moves=${DLGM}"
            COMM=$COMM" --use_dialogue=${DLG}"
            COMM=$COMM" --pov=${POV}"
            COMM=$COMM" --experiment=${EXP}"
            COMM=$COMM" --seq_model=${MODEL}"
            COMM=$COMM" --plan=Yes"
            COMM=$COMM" --save_path=${FOLDER}/${FILE_NAME}.torch"
            echo $(date +%F\ %T) $COMM" > ${FOLDER}/${FILE_NAME}.log"
            python3 $COMM > ${FOLDER}/${FILE_NAME}.log

            DLG="Yes"
            POV="None"

            FILE_NAME="gt_dialogue_moves_${MODEL}_dlgMove_${DLGM}_dlg_${DLG}_pov_${POV}_exp${EXP}_seed_${SEED}"
            COMM="baselines_with_dialogue_moves.py"
            COMM=$COMM" --device=${CUDA_DEVICE}"
            COMM=$COMM" --seed=${SEED}"
            COMM=$COMM" --use_dialogue_moves=${DLGM}"
            COMM=$COMM" --use_dialogue=${DLG}"
            COMM=$COMM" --pov=${POV}"
            COMM=$COMM" --experiment=${EXP}"
            COMM=$COMM" --seq_model=${MODEL}"
            COMM=$COMM" --plan=Yes"
            COMM=$COMM" --model_path=${FOLDER}/gt_dialogue_moves_${MODEL}_dlgMove_${DLGM}_dlg_No_pov_None_exp${EXP}_seed_${SEED}.torch"
            COMM=$COMM" --save_path=${FOLDER}/${FILE_NAME}.torch"
            echo $(date +%F\ %T) $COMM" > ${FOLDER}/${FILE_NAME}.log"
            python3 $COMM > ${FOLDER}/${FILE_NAME}.log

            DLG="No"
            POV="First"

            FILE_NAME="gt_dialogue_moves_${MODEL}_dlgMove_${DLGM}_dlg_${DLG}_pov_${POV}_exp${EXP}_seed_${SEED}"
            COMM="baselines_with_dialogue_moves.py"
            COMM=$COMM" --device=${CUDA_DEVICE}"
            COMM=$COMM" --seed=${SEED}"
            COMM=$COMM" --use_dialogue_moves=${DLGM}"
            COMM=$COMM" --use_dialogue=${DLG}"
            COMM=$COMM" --pov=${POV}"
            COMM=$COMM" --experiment=${EXP}"
            COMM=$COMM" --seq_model=${MODEL}"
            COMM=$COMM" --plan=Yes"
            COMM=$COMM" --model_path=${FOLDER}/gt_dialogue_moves_${MODEL}_dlgMove_${DLGM}_dlg_No_pov_None_exp${EXP}_seed_${SEED}.torch"
            COMM=$COMM" --save_path=${FOLDER}/${FILE_NAME}.torch"
            echo $(date +%F\ %T) $COMM" > ${FOLDER}/${FILE_NAME}.log"
            python3 $COMM > ${FOLDER}/${FILE_NAME}.log

            DLG="Yes"
            POV="First"

            FILE_NAME="gt_dialogue_moves_${MODEL}_dlgMove_${DLGM}_dlg_${DLG}_pov_${POV}_exp${EXP}_seed_${SEED}_DlgFirst"
            COMM="baselines_with_dialogue_moves.py"
            COMM=$COMM" --device=${CUDA_DEVICE}"
            COMM=$COMM" --seed=${SEED}"
            COMM=$COMM" --use_dialogue_moves=${DLGM}"
            COMM=$COMM" --use_dialogue=${DLG}"
            COMM=$COMM" --pov=${POV}"
            COMM=$COMM" --experiment=${EXP}"
            COMM=$COMM" --seq_model=${MODEL}"
            COMM=$COMM" --plan=Yes"
            COMM=$COMM" --model_path=${FOLDER}/gt_dialogue_moves_${MODEL}_dlgMove_${DLGM}_dlg_Yes_pov_None_exp${EXP}_seed_${SEED}.torch"
            COMM=$COMM" --save_path=${FOLDER}/${FILE_NAME}.torch"
            echo $(date +%F\ %T) $COMM" > ${FOLDER}/${FILE_NAME}.log"
            python3 $COMM > ${FOLDER}/${FILE_NAME}.log

            DLG="Yes"
            POV="First"

            FILE_NAME="gt_dialogue_moves_${MODEL}_dlgMove_${DLGM}_dlg_${DLG}_pov_${POV}_exp${EXP}_seed_${SEED}_VidFirst"
            COMM="baselines_with_dialogue_moves.py"
            COMM=$COMM" --device=${CUDA_DEVICE}"
            COMM=$COMM" --seed=${SEED}"
            COMM=$COMM" --use_dialogue_moves=${DLGM}"
            COMM=$COMM" --use_dialogue=${DLG}"
            COMM=$COMM" --pov=${POV}"
            COMM=$COMM" --experiment=${EXP}"
            COMM=$COMM" --seq_model=${MODEL}"
            COMM=$COMM" --plan=Yes"
            COMM=$COMM" --model_path=${FOLDER}/gt_dialogue_moves_${MODEL}_dlgMove_${DLGM}_dlg_No_pov_First_exp${EXP}_seed_${SEED}.torch"
            COMM=$COMM" --save_path=${FOLDER}/${FILE_NAME}.torch"
            echo $(date +%F\ %T) $COMM" > ${FOLDER}/${FILE_NAME}.log"
            python3 $COMM > ${FOLDER}/${FILE_NAME}.log

        done
    done
done


echo "Done!"

