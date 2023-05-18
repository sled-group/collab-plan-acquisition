echo >> plan_stab.txt
echo >> plan_stab.txt
echo 2  >> plan_stab.txt
python3 plan_stab.py --use_dialogue=No --use_dialogue_moves=No --use_dialogue=No --experiment=3 --seq_model=LSTM --pov=First --plan=Yes --intermediate=0 --root_path=models/incremental_pretrained_2/ >> plan_stab.txt
# echo >> plan_stab.txt
# echo >> plan_stab.txt
# echo 3 >> plan_stab.txt
# python3 plan_stab.py --use_dialogue=No --use_dialogue_moves=No --use_dialogue=No --experiment=3 --seq_model=LSTM --pov=First --plan=Yes --intermediate=0 --root_path=models/incremental_pretrained_3/ >> plan_stab.txt
