echo $$
# calc 3.5*60*60 | xargs sleep

# for POV in 'None' 'First'
# python3 dialogue_act_classifier.py --seed=Fixed --use_dialogue=Yes --experiment=0 --seq_model=GRU            --pov=None  --plan=Yes --save_path=dialogue_act_None_GRU.torch          > dialogue_act_None_GRU.log
python3 dialogue_act_classifier.py --seed=Fixed --use_dialogue=Yes --experiment=0 --seq_model=LSTM           --pov=None  --plan=Yes --save_path=dialogue_act_None_LSTM.torch         > dialogue_act_None_LSTM.log
python3 dialogue_act_classifier.py --seed=Fixed --use_dialogue=Yes --experiment=0 --seq_model=Transformer    --pov=None  --plan=Yes --save_path=dialogue_act_None_Transformer.torch  > dialogue_act_None_Transformer.log
# python3 dialogue_act_classifier.py --seed=Fixed --use_dialogue=Yes --experiment=0 --seq_model=GRU            --pov=First --plan=Yes --save_path=dialogue_act_First_GRU.torch         > dialogue_act_First_GRU.log
python3 dialogue_act_classifier.py --seed=Fixed --use_dialogue=Yes --experiment=0 --seq_model=LSTM           --pov=First --plan=Yes --save_path=dialogue_act_First_LSTM.torch        > dialogue_act_First_LSTM.log
python3 dialogue_act_classifier.py --seed=Fixed --use_dialogue=Yes --experiment=0 --seq_model=Transformer    --pov=First --plan=Yes --save_path=dialogue_act_First_Transformer.torch > dialogue_act_First_Transformer.log

# python3 dialogue_act_classifier.py --seed=Fixed --use_dialogue=Yes --experiment=0 --seq_model=GRU            --pov=None  --plan=No  --save_path=dialogue_act_None_GRU_no_plan.torch           > dialogue_act_None_GRU_no_plan.log
python3 dialogue_act_classifier.py --seed=Fixed --use_dialogue=Yes --experiment=0 --seq_model=LSTM           --pov=None  --plan=No  --save_path=dialogue_act_None_LSTM_no_plan.torch          > dialogue_act_None_LSTM_no_plan.log
python3 dialogue_act_classifier.py --seed=Fixed --use_dialogue=Yes --experiment=0 --seq_model=Transformer    --pov=None  --plan=No  --save_path=dialogue_act_None_Transformer_no_plan.torch   > dialogue_act_None_Transformer_no_plan.log
# python3 dialogue_act_classifier.py --seed=Fixed --use_dialogue=Yes --experiment=0 --seq_model=GRU            --pov=First --plan=No  --save_path=dialogue_act_First_GRU_no_plan.torch          > dialogue_act_First_GRU_no_plan.log
python3 dialogue_act_classifier.py --seed=Fixed --use_dialogue=Yes --experiment=0 --seq_model=LSTM           --pov=First --plan=No  --save_path=dialogue_act_First_LSTM_no_plan.torch         > dialogue_act_First_LSTM_no_plan.log
python3 dialogue_act_classifier.py --seed=Fixed --use_dialogue=Yes --experiment=0 --seq_model=Transformer    --pov=First --plan=No  --save_path=dialogue_act_First_Transformer_no_plan.torch  > dialogue_act_First_Transformer_no_plan.log
