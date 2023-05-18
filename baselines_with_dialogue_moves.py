from glob import glob
import os, json, sys
import torch, random, torch.nn as nn, numpy as np
from torch import optim
from random import shuffle
from sklearn.metrics import accuracy_score, f1_score
from src.data.game_parser import GameParser, make_splits, onehot, DEVICE, set_seed
from src.models.model_with_dialogue_moves import Model
import argparse

def print_epoch(data,acc_loss,lst):
    print(f'{acc_loss/len(lst):9.4f}',end='; ',flush=True)
    data = list(zip(*data))
    for x in data:
        a, b = list(zip(*x))
        if max(a) <= 1:
            print(f'({accuracy_score(a,b):5.3f},{f1_score(a,b,average="weighted"):5.3f},{sum(a)/len(a):5.3f},{sum(b)/len(b):5.3f},{len(b)})', end=' ',flush=True)
        else:
            print(f'({accuracy_score(a,b):5.3f},{f1_score(a,b,average="weighted"):5.3f},{len(b)})', end=' ',flush=True)
    print('', end='; ',flush=True)

def do_split(model,lst,exp,criterion,optimizer=None,global_plan=False, player_plan=False,device=DEVICE):
    data = []
    acc_loss = 0
    for game in lst:
        l = model(game, global_plan=global_plan, player_plan=player_plan)
        prediction = []
        ground_truth = []
        for gt, prd in l:
            lbls = [int(a==b) for a,b in zip(gt[0],gt[1])]
            lbls += [['NO', 'MAYBE', 'YES'].index(gt[0][0]),['NO', 'MAYBE', 'YES'].index(gt[0][1])]
            if gt[0][2] in game.materials_dict:
                lbls.append(game.materials_dict[gt[0][2]])
            else:
                lbls.append(0)
            lbls += [['NO', 'MAYBE', 'YES'].index(gt[1][0]),['NO', 'MAYBE', 'YES'].index(gt[1][1])]
            if gt[1][2] in game.materials_dict:
                lbls.append(game.materials_dict[gt[1][2]])
            else:
                lbls.append(0)
            prd = prd[exp:exp+1]
            lbls = lbls[exp:exp+1]
            data.append([(g,torch.argmax(p).item()) for p,g in zip(prd,lbls)])
            # p, g = zip(*[(p,torch.eye(p.shape[0]).float()[g]) for p,g in zip(prd,lbls)])
            if exp == 0:
                pairs = list(zip(*[(pr,gt) for pr,gt in zip(prd,lbls) if gt==0 or (random.random() < 2/3)]))
            elif exp == 1:
                pairs = list(zip(*[(pr,gt) for pr,gt in zip(prd,lbls) if gt==0 or (random.random() < 5/6)]))
            elif exp == 2:
                pairs = list(zip(*[(pr,gt) for pr,gt in zip(prd,lbls) if gt==1 or (random.random() < 5/6)]))
            else:
                pairs = list(zip(*[(pr,gt) for pr,gt in zip(prd,lbls)]))
            # print(pairs)
            if pairs:
                p,g = pairs
            else:
                continue
            # print(p,g)
            prediction.append(torch.cat(p))
            
            # ground_truth.append(torch.cat(g))
            ground_truth += g
            
        if prediction:
            prediction = torch.stack(prediction)
        else:
            continue
        if ground_truth:
            # ground_truth = torch.stack(ground_truth).float().to(DEVICE)
            ground_truth = torch.tensor(ground_truth).long().to(device)
        else:
            continue
            
            
        loss = criterion(prediction,ground_truth)
        
        if model.training and (not optimizer is None):
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 10)
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        acc_loss += loss.item()
        # return data, acc_loss + loss.item()
    print_epoch(data,acc_loss,lst)
    return acc_loss, data

def main(args):
    print(args, flush=True)
    print(f'PID: {os.getpid():6d}', flush=True)

    if isinstance(args.device, int) and args.device >= 0:
        DEVICE = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        print(f'Using {DEVICE}')
    else:
        print('Device must be a zero or positive integer, but got',args.device)
        exit()
    
    if isinstance(args.seed, int) and args.seed >= 0:
        seed = set_seed(args.seed)
    else:
        print('Seed must be a zero or positive integer, but got',args.seed)
        exit()
    
    # dataset_splits = make_splits('config/dataset_splits.json')
    # dataset_splits = make_splits('config/dataset_splits_dev.json')
    # dataset_splits = make_splits('config/dataset_splits_old.json')
    dataset_splits = make_splits('config/dataset_splits_new.json')
    
    if args.use_dialogue=='Yes':
        d_flag = True
    elif args.use_dialogue=='No':
        d_flag = False
    else:
        print('Use dialogue must be in [Yes, No], but got',args.use_dialogue)
        exit()
    
    if args.use_dialogue_moves=='Yes':
        d_move_flag = True
    elif args.use_dialogue_moves=='No':
        d_move_flag = False
    else:
        print('Use dialogue must be in [Yes, No], but got',args.use_dialogue)
        exit()
        
    if not args.experiment in list(range(9)):
        print('Experiment must be in',list(range(9)),', but got',args.experiment)
        exit()
        

    if args.seq_model=='GRU':
        seq_model = 0
    elif args.seq_model=='LSTM':
        seq_model = 1
    elif args.seq_model=='Transformer':
        seq_model = 2
    else:
        print('The sequence model must be in [GRU, LSTM, Transformer], but got', args.seq_model)
        exit()

    if args.plans=='Yes':
        global_plan = (args.pov=='Third') or ((args.pov=='None') and (args.experiment in list(range(3))))
        player_plan = (args.pov=='First') or ((args.pov=='None') and (args.experiment in list(range(3,9))))
    elif args.plans=='No' or args.plans is None:
        global_plan = False
        player_plan = False
    else:
        print('Use Plan must be in [Yes, No], but got',args.plan)
        exit()
    print('global_plan', global_plan, 'player_plan', player_plan)
    
    if args.pov=='None':
        val    = [GameParser(f,d_flag,0,0,d_move_flag) for f in dataset_splits['validation']]
        train  = [GameParser(f,d_flag,0,0,d_move_flag) for f in dataset_splits['training']]
        if args.experiment > 2:
            val   += [GameParser(f,d_flag,4,0,d_move_flag) for f in dataset_splits['validation']]
            train += [GameParser(f,d_flag,4,0,d_move_flag) for f in dataset_splits['training']]
    elif args.pov=='Third':
        val    = [GameParser(f,d_flag,3,0,d_move_flag) for f in dataset_splits['validation']]
        train  = [GameParser(f,d_flag,3,0,d_move_flag) for f in dataset_splits['training']]
    elif args.pov=='First':
        val    = [GameParser(f,d_flag,1,0,d_move_flag) for f in dataset_splits['validation']]
        train  = [GameParser(f,d_flag,1,0,d_move_flag) for f in dataset_splits['training']]
        val   += [GameParser(f,d_flag,2,0,d_move_flag) for f in dataset_splits['validation']]
        train += [GameParser(f,d_flag,2,0,d_move_flag) for f in dataset_splits['training']]
    else:
        print('POV must be in [None, First, Third], but got', args.pov)
        exit()

    model = Model(seq_model,DEVICE).to(DEVICE)

    print(model)
    model.train()

    learning_rate = 1e-4
    num_epochs = 1000#2#1#
    weight_decay=1e-4

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    # optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    # optimizer = optim.Adadelta(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()

    print(str(criterion), str(optimizer))

    min_acc_loss = 100
    max_f1 = 0
    epochs_since_improvement = 0
    wait_epoch = 100

    if args.model_path is not None:
        print(f'Loading {args.model_path}')
        model.load_state_dict(torch.load(args.model_path))
        acc_loss, data = do_split(model,val,args.experiment,criterion,global_plan=global_plan, player_plan=player_plan, device=DEVICE)
        data = list(zip(*data))
        for x in data:
            a, b = list(zip(*x))
        f1 = f1_score(a,b,average='weighted')
        f1 = f1_score(a,b,average='weighted')
        if (max_f1 < f1):
            max_f1 = f1
            epochs_since_improvement = 0
            print('^')
            torch.save(model.cpu().state_dict(), args.save_path)
            model = model.to(DEVICE)
    else:
        print('Training model from scratch', flush=True)
    # exit()

    for epoch in range(num_epochs):
        print(f'{os.getpid():6d} {epoch+1:4d},',end=' ', flush=True)
        shuffle(train)
        model.train()
        do_split(model,train,args.experiment,criterion,optimizer=optimizer,global_plan=global_plan, player_plan=player_plan, device=DEVICE)
        model.eval()
        acc_loss, data = do_split(model,val,args.experiment,criterion,global_plan=global_plan, player_plan=player_plan, device=DEVICE)
        
        data = list(zip(*data))
        for x in data:
            a, b = list(zip(*x))
        f1 = f1_score(a,b,average='weighted')
        if (max_f1 < f1):
            max_f1 = f1
            epochs_since_improvement = 0
            print('^')
            torch.save(model.cpu().state_dict(), args.save_path)
            model = model.to(DEVICE)
        else:
            epochs_since_improvement += 1
            print()
        # if (min_acc_loss > acc_loss):
        #     min_acc_loss = acc_loss
        #     epochs_since_improvement = 0
        #     print('^')
        # else:
        #     epochs_since_improvement += 1
        #     print()
            
        if epoch > wait_epoch and epochs_since_improvement > 20:
            break
    print()
    print('Test')
    model.load_state_dict(torch.load(args.save_path))

    val = None
    train = None
    if args.pov=='None':
        test = [GameParser(f,d_flag,0,0,d_move_flag) for f in dataset_splits['test']]
        if args.experiment > 2:
            test += [GameParser(f,d_flag,4,0,d_move_flag) for f in dataset_splits['test']]
    elif args.pov=='Third':
        test = [GameParser(f,d_flag,3,0,d_move_flag) for f in dataset_splits['test']]
    elif args.pov=='First':
        test  = [GameParser(f,d_flag,1,0,d_move_flag) for f in dataset_splits['test']]
        test += [GameParser(f,d_flag,2,0,d_move_flag) for f in dataset_splits['test']]
    else:
        print('POV must be in [None, First, Third], but got', args.pov)
    model.eval()
    _, data = do_split(model,test,args.experiment,criterion,global_plan=global_plan, player_plan=player_plan, device=DEVICE)
    
    print()
    print(data)
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--pov', type=str, 
                    help='point of view [None, First, Third]')
    parser.add_argument('--use_dialogue', type=str, 
                    help='Use dialogue [Yes, No]')
    parser.add_argument('--use_dialogue_moves', type=str, 
                    help='Use dialogue [Yes, No]')
    parser.add_argument('--plans', type=str, 
                    help='Use dialogue [Yes, No]')
    parser.add_argument('--seq_model', type=str, 
                    help='point of view [GRU, LSTM, Transformer]')
    parser.add_argument('--experiment', type=int, 
                    help='point of view [0:AggQ1, 1:AggQ2, 2:AggQ3, 3:P0Q1, 4:P0Q2, 5:P0Q3, 6:P1Q1, 7:P1Q2, 8:P1Q3]')
    parser.add_argument('--seed', type=int, 
                    help='Selet random seed by index [0, 1, 2, ...]. 0 -> random seed set to 0. n>0 -> random seed '
                    'set to n\'th random number with original seed set to 0')
    parser.add_argument('--save_path', type=str, 
                    help='path where to save model')
    parser.add_argument('--model_path', type=str, default=None,
                    help='path to the pretrained model to be loaded')
    parser.add_argument('--device', type=int, default=0,
                    help='select cuda device number')
    
    main(parser.parse_args())