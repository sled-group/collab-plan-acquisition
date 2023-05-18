FOLDER="models/gt_dialogue_moves_bootstrap"
# FOLDER="models/gt_dialogue_moves"
# FOLDER="models/gt_dialogue_moves_bootstrap"
# FOLDER2="models/gt_dialogue_moves"
# FOLDER2="models/gt_dialogue_moves_bootstrap"
EXP=$1
# for MODEL in Transformer; do # LSTM; do # LSTM Transformer; do # 
#     for DLG in No Yes; do
#         for POV in First None; do
#             for SEED in 0 1 2 3 4 5; do # 2 3; do
#                 FILE_NAME="gt_dialogue_moves_${MODEL}_dlgMove_No_dlg_${DLG}_pov_${POV}_exp${EXP}_seed_${SEED}.log"
#                 printf "${FOLDER}/${FILE_NAME} "
#                 printf $(tail -3 "${FOLDER}/${FILE_NAME}" | head -1 | cut -d ';' -f 2)
#                 tail -3 "${FOLDER2}/gt_dialogue_moves_${MODEL}_dlgMove_Yes_dlg_${DLG}_pov_${POV}_exp${EXP}_seed_${SEED}.log" | head -1 | cut -d ';' -f 2
#             done
#         done
#     done
# done

FOLDER="models/gt_dialogue_moves"

for file in $FOLDER/*Transformer*_dlgMove_No_*_exp$EXP*log; do
    printf $file' '
    printf $(tail -3 "${file}" | head -1 | cut -d ';' -f 2)
    tail -3 ${file/dlgMove_No/dlgMove_Yes} | head -1 | cut -d ';' -f 2
done

for file in $FOLDER/*Transformer*_dlgMove_Yes_dlg_No_pov_None_exp$EXP*log; do
    printf $file' '
    printf $(tail -3 "${file}" | head -1 | cut -d ';' -f 2)
    echo
    # tail -3 ${file/dlgMove_No/dlgMove_Yes} | head -1 | cut -d ';' -f 2
done

FOLDER="models/gt_dialogue_moves_bootstrap"

for file in $FOLDER/*Transformer*_dlgMove_No_*_exp$EXP*log; do
    printf $file' '
    printf $(tail -3 "${file}" | head -1 | cut -d ';' -f 2)
    tail -3 ${file/dlgMove_No/dlgMove_Yes} | head -1 | cut -d ';' -f 2
done

for file in $FOLDER/*Transformer*_dlgMove_Yes_dlg_No_pov_None_exp$EXP*log; do
    printf $file' '
    printf $(tail -3 "${file}" | head -1 | cut -d ';' -f 2)
    echo
    # tail -3 ${file/dlgMove_No/dlgMove_Yes} | head -1 | cut -d ';' -f 2
done

FOLDER="models/gt_dialogue_moves_bootstrap_DlgMoveFirst"

for file in $FOLDER/*Transformer*_dlgMove_No_*_exp$EXP*log; do
    printf $file' '
    printf $(tail -3 "${file}" | head -1 | cut -d ';' -f 2)
    tail -3 ${file/dlgMove_No/dlgMove_Yes} | head -1 | cut -d ';' -f 2
done

for file in $FOLDER/*Transformer*_dlgMove_Yes_dlg_No_pov_None_exp$EXP*log; do
    printf $file' '
    printf $(tail -3 "${file}" | head -1 | cut -d ';' -f 2)
    echo
    # tail -3 ${file/dlgMove_No/dlgMove_Yes} | head -1 | cut -d ';' -f 2
done

