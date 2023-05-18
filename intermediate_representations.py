from src.models.model_with_dialogue_moves import Model as ToMModel
from src.models.dialogue_act_classification_model import Model as DActModel
from src.data.game_parser import GameParser
import torch
import argparse
import json
import os
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_TYPES = {
    'GRU' : 0,
    'LSTM' : 1,
    'Transformer' : 2,
}

def main(args):
    # model_file = "models/ToM_models/dialogue_No_pov_First_Transformer_6_a.torch"
    model_file = "models/gt_dialogue_moves_bootstrap_DlgMoveFirst/gt_dialogue_moves_Transformer_dlgMove_No_dlg_Yes_pov_First_exp6_seed_2_VidFirst.torch"
    use_dialogue = "Yes" #model_file.split("_dlg_")[-1].split["_"][0]
    model_type_name = "Transformer"#model_file.split("_")[-3]
    model_type = MODEL_TYPES[model_type_name]
    model = ToMModel(model_type).to(DEVICE)
    model.load_state_dict(torch.load(model_file))
    dataset_splits = json.load(open('config/dataset_splits.json'))
    for set in dataset_splits.values():
        for path in set:
            for pov in [1, 2]:
                out_file = f'{path}/intermediate_ToM6_{path.split("/")[-1]}_player{pov}.npz'
                if os.path.isfile(out_file):
                    continue
                game = GameParser(path,use_dialogue=='Yes',pov,0,True)
                l = model(game, global_plan=False, player_plan=True,intermediate=True).cpu().data.numpy()
                np.savez_compressed(open(out_file,'wb'), data=l)
                print(out_file,l.shape,model_type_name,use_dialogue,use_dialogue=='Yes')
                # break
        #     break
        # break
                
    
    # model_file = "models/ToM_models/dialogue_No_pov_First_Transformer_7_i.torch"
    model_file = "models/gt_dialogue_moves_bootstrap_DlgMoveFirst/gt_dialogue_moves_Transformer_dlgMove_Yes_dlg_No_pov_None_exp7_seed_5.torch"
    use_dialogue = "No"#model_file.split("_")[-6]
    model_type_name = 'Transformer'#model_file.split("_")[-3]
    model_type = MODEL_TYPES[model_type_name]
    model = ToMModel(model_type).to(DEVICE)
    model.load_state_dict(torch.load(model_file))
    dataset_splits = json.load(open('config/dataset_splits.json'))
    for set in dataset_splits.values():
        for path in set:
            for pov in [1, 2]:
                out_file = f'{path}/intermediate_ToM7_{path.split("/")[-1]}_player{pov}.npz'
                if os.path.isfile(out_file):
                    continue
                game = GameParser(path,use_dialogue=='Yes',4,0,True)
                l = model(game, global_plan=False, player_plan=True,intermediate=True).cpu().data.numpy()
                np.savez_compressed(open(out_file,'wb'), data=l)
                print(out_file,l.shape,model_type_name,use_dialogue,use_dialogue=='Yes')
                # break
        #     break
        # break
                
    
    # model_file = "models/ToM_models/dialogue_Yes_pov_First_LSTM_8_j.torch"
    model_file = "models/gt_dialogue_moves_bootstrap_DlgMoveFirst/gt_dialogue_moves_Transformer_dlgMove_Yes_dlg_No_pov_None_exp8_seed_5.torch"
    use_dialogue = "No"#model_file.split("_")[-6]
    model_type_name = 'Transformer'#model_file.split("_")[-3]
    model_type = MODEL_TYPES[model_type_name]
    model = ToMModel(model_type).to(DEVICE)
    model.load_state_dict(torch.load(model_file))
    dataset_splits = json.load(open('config/dataset_splits.json'))
    for set in dataset_splits.values():
        for path in set:
            for pov in [1, 2]:
                out_file = f'{path}/intermediate_ToM8_{path.split("/")[-1]}_player{pov}.npz'
                if os.path.isfile(out_file):
                    continue
                game = GameParser(path,use_dialogue=='Yes',4,True)
                l = model(game, global_plan=False, player_plan=True,intermediate=True).cpu().data.numpy()
                np.savez_compressed(open(out_file,'wb'), data=l)
                print(out_file,l.shape,model_type_name,use_dialogue,use_dialogue=='Yes')
                # break
        #     break
        # break
                
    
    # model_file = "models/20211230/dialogue_act_First_LSTM.torch"
    # use_dialogue = True
    # model_type_name = model_file.split("_")[-1].split('.')[0]
    # model_type = MODEL_TYPES[model_type_name]
    # model = DActModel(model_type).to(DEVICE)
    # model.load_state_dict(torch.load(model_file))
    # dataset_splits = json.load(open('config/dataset_splits.json'))
    # for set in dataset_splits.values():
    #     for path in set:
    #         for pov in [1, 2]:
    #             out_file = f'{path}/intermediate_DAct_{path.split("/")[-1]}_player{pov}.npz'
    #             if os.path.isfile(out_file):
    #                 continue
    #             game = GameParser(path,use_dialogue=='Yes',pov)
    #             l = model(game, global_plan=False, player_plan=True,intermediate=True).cpu().data.numpy()
    #             np.savez_compressed(open(out_file,'wb'), data=l)
    #             print(out_file,l.shape,model_type_name,use_dialogue)
    #             # break
    #     #     break
    #     # break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--pov', type=str, 
                    help='point of view [None, First, Third]')
    parser.add_argument('--use_dialogue', type=str, 
                    help='Use dialogue [Yes, No]')
    parser.add_argument('--plans', type=str, 
                    help='Use dialogue [Yes, No]')
    parser.add_argument('--seq_model', type=str, 
                    help='point of view [GRU, LSTM, Transformer, None]')
    parser.add_argument('--experiment', type=int, 
                    help='point of view [0:AggQ1, 1:AggQ2, 2:AggQ3, 3:P0Q1, 4:P0Q2, 5:P0Q3, 6:P1Q1, 7:P1Q2, 8:P1Q3]')
    parser.add_argument('--save_path', type=str, 
                    help='path where to save model')
    parser.add_argument('--seed', type=str, 
                    help='Use random or fixed seed [Random, Fixed]')
    
    main(parser.parse_args())