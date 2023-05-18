echo >> plan_move_plots.txt
echo >> plan_move_plots.txt
echo 2  >> plan_move_plots.txt
python3 plan_move_plots.py --use_dialogue=No --use_dialogue_moves=No --use_dialogue=No --experiment=3 --seq_model=LSTM --pov=First --plan=Yes --intermediate=0 --root_path=models/incremental_pretrained_2/ --save_path=models/incremental_pretrained_2/plan_exp3_LSTM_dlg_No_move_No_int0_seed_0.torch >> plan_move_plots.txt 

echo >> plan_move_plots.txt
echo >> plan_move_plots.txt
echo 3  >> plan_move_plots.txt
python3 plan_move_plots.py --use_dialogue=No --use_dialogue_moves=No --use_dialogue=No --experiment=3 --seq_model=LSTM --pov=First --plan=Yes --intermediate=0 --root_path=models/incremental_pretrained_3/ --save_path=models/incremental_pretrained_3/plan_exp3_LSTM_dlg_No_move_No_int0_seed_0.torch >> plan_move_plots.txt 
