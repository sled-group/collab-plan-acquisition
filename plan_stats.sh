
# for INT in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
#     # printf plan_exp3_LSTM_int$INT.torch
#     python3 plan_stats.py --seed=Fixed --use_dialogue=Yes --experiment=3 --seq_model=LSTM --pov=First --plan=Yes --intermediate=$INT --save_path=$1/plan_exp3_LSTM_int$INT_seed_0.torch
# done


        # FILE_NAME="plan_exp${EXP}_${MODEL}_int${INT}_seed_${SEED}"
        # COMM="plan_predictor.py --seed=${SEED} --use_dialogue=Yes"
        # COMM=$COMM" --device=${CUDA_DEVICE}"
        # COMM=$COMM" --seed=${SEED}"
        # COMM=$COMM" --use_dialogue_moves=No"
        # COMM=$COMM" --use_dialogue=No"
        # COMM=$COMM" --experiment=${EXP}"
        # COMM=$COMM" --seq_model=${MODEL}"
        # COMM=$COMM" --pov=First"
        # COMM=$COMM" --plan=Yes"
        # COMM=$COMM" --intermediate=${INT}"
        # COMM=$COMM" --save_path=${FOLDER}/${FILE_NAME}.torch"

echo plan_vid_only
python3 plan_stats.py --use_dialogue=No --use_dialogue_moves=No --experiment=3 --seq_model=LSTM --pov=First --plan=Yes --intermediate=0 --save_path=models/plan_vid_only/plan_exp3_LSTM_int0_seed_0.torch
echo plan_no_move
python3 plan_stats.py --use_dialogue=Yes --use_dialogue_moves=No --experiment=3 --seq_model=LSTM --pov=First --plan=Yes --intermediate=0 --save_path=models/plan_no_move/plan_exp3_LSTM_int0_seed_0.torch
echo plan_no_dlg
python3 plan_stats.py --use_dialogue=No --use_dialogue_moves=Yes --experiment=3 --seq_model=LSTM --pov=First --plan=Yes --intermediate=0 --save_path=models/plan_no_dlg/plan_exp3_LSTM_int0_seed_0.torch
echo plan_all_inputs
python3 plan_stats.py --use_dialogue=Yes --use_dialogue_moves=Yes --experiment=3 --seq_model=LSTM --pov=First --plan=Yes --intermediate=0 --save_path=models/plan_all_inputs/plan_exp3_LSTM_int0_seed_0.torch