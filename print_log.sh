for EXP in 3; do # 2 3; do
    for DLG in "No" "Yes"; do
        for DMOVE in "No" "Yes"; do
            for INT in 0 1 2 3 4 5 6 7; do
                FILE_NAME="plan_exp${EXP}_Transformer_dlg_${DLG}_move_${DMOVE}_int${INT}_seed_*"
                for f in models/incremental_pretrained/plan_exp${EXP}_Transformer_dlg_${DLG}_move_${DMOVE}_int${INT}_seed_*.log; do
                    printf $f
                    cat $f | grep Test -A 1 | tail -1
                done
                echo                
            done
        done
    done
done