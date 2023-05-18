echo "dialogue move slassification script" $$
# calc 3.5*60*60 | xargs sleep

DEST_DIR='models/dlg_move_cls_move_only_3_rev_weights'

mkdir -p $DEST_DIR

# for POV in 'None' 'First'

# python3 dialogue_move_classifier.py --seed=0 --use_dialogue=Yes --experiment=0 --seq_model=GRU            --pov=First --plan=Yes --save_path=$DEST_DIR/dialogue_move_First_GRU.torch         > $DEST_DIR/dialogue_move_First_GRU.log
python3 dialogue_move_classifier.py --seed=0 --use_dialogue=Yes --experiment=0 --seq_model=LSTM           --pov=First --plan=Yes --save_path=$DEST_DIR/dialogue_move_First_LSTM.torch        > $DEST_DIR/dialogue_move_First_LSTM.log
# python3 dialogue_move_classifier.py --seed=0 --use_dialogue=Yes --experiment=0 --seq_model=Transformer    --pov=First --plan=Yes --save_path=$DEST_DIR/dialogue_move_First_Transformer.torch > $DEST_DIR/dialogue_move_First_Transformer.log
# python3 dialogue_move_classifier.py --seed=0 --use_dialogue=Yes --experiment=0 --seq_model=GRU            --pov=None  --plan=Yes --save_path=$DEST_DIR/dialogue_move_None_GRU.torch          > $DEST_DIR/dialogue_move_None_GRU.log
python3 dialogue_move_classifier.py --seed=0 --use_dialogue=Yes --experiment=0 --seq_model=LSTM           --pov=None  --plan=Yes --save_path=$DEST_DIR/dialogue_move_None_LSTM.torch         > $DEST_DIR/dialogue_move_None_LSTM.log
# python3 dialogue_move_classifier.py --seed=0 --use_dialogue=Yes --experiment=0 --seq_model=Transformer    --pov=None  --plan=Yes --save_path=$DEST_DIR/dialogue_move_None_Transformer.torch  > $DEST_DIR/dialogue_move_None_Transformer.log

# python3 dialogue_move_classifier.py --seed=0 --use_dialogue=Yes --experiment=0 --seq_model=GRU            --pov=None  --plan=No  --save_path=$DEST_DIR/dialogue_move_None_GRU_no_plan.torch           > $DEST_DIR/dialogue_move_None_GRU_no_plan.log
python3 dialogue_move_classifier.py --seed=0 --use_dialogue=Yes --experiment=0 --seq_model=LSTM           --pov=None  --plan=No  --save_path=$DEST_DIR/dialogue_move_None_LSTM_no_plan.torch          > $DEST_DIR/dialogue_move_None_LSTM_no_plan.log
# python3 dialogue_move_classifier.py --seed=0 --use_dialogue=Yes --experiment=0 --seq_model=Transformer    --pov=None  --plan=No  --save_path=$DEST_DIR/dialogue_move_None_Transformer_no_plan.torch   > $DEST_DIR/dialogue_move_None_Transformer_no_plan.log
# python3 dialogue_move_classifier.py --seed=0 --use_dialogue=Yes --experiment=0 --seq_model=GRU            --pov=First --plan=No  --save_path=$DEST_DIR/dialogue_move_First_GRU_no_plan.torch          > $DEST_DIR/dialogue_move_First_GRU_no_plan.log
python3 dialogue_move_classifier.py --seed=0 --use_dialogue=Yes --experiment=0 --seq_model=LSTM           --pov=First --plan=No  --save_path=$DEST_DIR/dialogue_move_First_LSTM_no_plan.torch         > $DEST_DIR/dialogue_move_First_LSTM_no_plan.log
# python3 dialogue_move_classifier.py --seed=0 --use_dialogue=Yes --experiment=0 --seq_model=Transformer    --pov=First --plan=No  --save_path=$DEST_DIR/dialogue_move_First_Transformer_no_plan.torch  > $DEST_DIR/dialogue_move_First_Transformer_no_plan.log
