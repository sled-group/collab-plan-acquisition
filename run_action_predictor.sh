echo $$
# calc 3.5*60*60 | xargs sleep

# python3 action_predictor.py --seed=Fixed --use_dialogue=Yes --experiment=0 --seq_model=GRU            --pov=First --plan=Yes --save_path=action_pred_First_GRU.torch         > action_pred_First_GRU.log
python3 action_predictor.py --seed=Fixed --use_dialogue=Yes --experiment=0 --seq_model=LSTM           --pov=First --plan=Yes --save_path=action_pred_First_LSTM.torch        > action_pred_First_LSTM.log
python3 action_predictor.py --seed=Fixed --use_dialogue=Yes --experiment=0 --seq_model=Transformer    --pov=First --plan=Yes --save_path=action_pred_First_Transformer.torch > action_pred_First_Transformer.log

# python3 action_predictor.py --seed=Fixed --use_dialogue=Yes --experiment=0 --seq_model=GRU            --pov=First --plan=No  --save_path=action_pred_First_GRU_no_plan.torch          > action_pred_First_GRU_no_plan.log
python3 action_predictor.py --seed=Fixed --use_dialogue=Yes --experiment=0 --seq_model=LSTM           --pov=First --plan=No  --save_path=action_pred_First_LSTM_no_plan.torch         > action_pred_First_LSTM_no_plan.log
python3 action_predictor.py --seed=Fixed --use_dialogue=Yes --experiment=0 --seq_model=Transformer    --pov=First --plan=No  --save_path=action_pred_First_Transformer_no_plan.torch  > action_pred_First_Transformer_no_plan.log



# # python3 action_predictor.py --seed=Fixed --use_dialogue=Yes --experiment=0 --seq_model=GRU            --pov=First --plan=Yes --save_path=action_pred_First_GRU.torch         > action_pred_First_GRU.log
# python3 action_predictor.py --seed=Fixed --use_dialogue=Yes --experiment=1 --seq_model=LSTM           --pov=First --plan=Yes --save_path=action_pred_First_LSTM1.torch        > action_pred_First_LSTM1.log
# python3 action_predictor.py --seed=Fixed --use_dialogue=Yes --experiment=1 --seq_model=Transformer    --pov=First --plan=Yes --save_path=action_pred_First_Transformer1.torch > action_pred_First_Transformer1.log

# # python3 action_predictor.py --seed=Fixed --use_dialogue=Yes --experiment=0 --seq_model=GRU            --pov=First --plan=No  --save_path=action_pred_First_GRU_no_plan.torch          > action_pred_First_GRU_no_plan.log
# python3 action_predictor.py --seed=Fixed --use_dialogue=Yes --experiment=1 --seq_model=LSTM           --pov=First --plan=No  --save_path=action_pred_First_LSTM_no_plan1.torch         > action_pred_First_LSTM_no_plan1.log
# python3 action_predictor.py --seed=Fixed --use_dialogue=Yes --experiment=1 --seq_model=Transformer    --pov=First --plan=No  --save_path=action_pred_First_Transformer_no_plan1.torch  > action_pred_First_Transformer_no_plan1.log
