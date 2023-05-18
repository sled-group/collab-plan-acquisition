from glob import glob
import os, json, sys
import torch, random, torch.nn as nn, numpy as np
from torch import optim
from random import shuffle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.data.game_parser import GameParser, make_splits, onehot, DEVICE, set_seed
from src.models.plan_model import Model
from src.models.losses import PlanLoss
import argparse

def print_epoch(data,acc_loss,lst,exp, incremental=False):
    print(f'{acc_loss:9.4f}',end='; ',flush=True)
    acc = []
    prec = []
    rec = []
    f1 = []
    iou = []
    total = []
    predicts = []
    targets = []
    # for x,game in zip(data,lst):
    for x in data:

        game = lst[x[2]]
        game_mats = game.plan['materials']
        pov_plan = game.plan[f'player{game.pov}']
        pov_plan_mat = game.__dict__[f'player{game.pov}_plan_mat']
        possible_mats = [game.materials_dict[x]-1 for x,_ in zip(game_mats[1:],pov_plan[1:])]
        possible_cand = [game.materials_dict[x]-1 for x,y in zip(game_mats[1:],pov_plan[1:]) if y['make'] and y['make'][0][0]==-1]
        possible_extra = [game.materials_dict[x]-1 for x,y in zip(game_mats[1:],pov_plan[1:]) if y['make'] and y['make'][0][0]>-1]
        a, b = x[:2]
        if exp == 3:
            a = a.reshape(21,21)
            for idx,aa in enumerate(a):
                if idx in possible_extra:
                    cand_idxs = set([i for i,x in enumerate(pov_plan_mat[idx]) if x])
                    th, _ = zip(*sorted([(i, x) for i, x in enumerate(aa) if i in possible_mats], key=lambda x:x[1])[-2:])
                    if len(cand_idxs.intersection(set(th))):
                        for jdx, _ in enumerate(aa):
                            a[idx,jdx] = pov_plan_mat[idx,jdx]
                    else:
                        for jdx, _ in enumerate(aa):
                            a[idx,jdx] = 0
                else:
                    for jdx, aaa in enumerate(aa):
                        a[idx,jdx] = 0
        elif exp == 2:
            a = a.reshape(21,21)
            for idx,aa in enumerate(a):
                if idx in possible_cand:
                    th = [x for i, x in enumerate(aa) if i in possible_mats]
                    th = sorted(th)
                    th = th[-2]
                    th = 1.1 if th < (1/21) else th
                    for jdx, aaa in enumerate(aa):
                        if idx in possible_mats:
                            a[idx,jdx] = 0 if aaa < th else 1
                        else:
                            a[idx,jdx] = 0
                else:
                    for jdx, aaa in enumerate(aa):
                        a[idx,jdx] = 0
                    
        else:
            a = a.reshape(21,21)
            for idx,aa in enumerate(a):
                th = sorted(aa)[-2]
                th = 1.1 if th < (2.1/21) else th
                for jdx, aaa in enumerate(aa):
                    a[idx,jdx] = 0 if aaa < th else 1
        a = a.reshape(-1)
        predicts.append(np.argmax(a))
        targets.append(np.argmax(a) if np.argmax(a) in [x for x in b if x] else np.argmax(b))
        acc.append(accuracy_score(a,b))
        sa = set([i for i,x in enumerate(a) if x])
        sb = set([i for i,x in enumerate(b) if x])
        i = len(sa.intersection(sb))
        u = len(sa.union(sb))
        if u > 0:
            a,b = zip(*[(x,y) for x,y in zip(a,b) if x+y > 0])
        f1.append(f1_score(b,a,zero_division=1))
        prec.append(precision_score(b,a,zero_division=1))
        rec.append(recall_score(b,a,zero_division=1))
        iou.append(i/u if u > 0 else 1)
        total.append(sum(a))
    print(
        #   f'({accuracy_score(targets,predicts):5.3f},'
        #   f'{np.mean(acc):5.3f},'
        #   f'{np.mean(prec):5.3f},'
        #   f'{np.mean(rec):5.3f},'
          f'{np.mean(f1):5.3f},'
          f'{np.mean(iou):5.3f},'
          f'{np.std(iou):5.3f},',
        #   f'{np.mean(total):5.3f})', 
          end=' ',flush=True)
    print('', end='; ',flush=True)
    return accuracy_score(targets,predicts), np.mean(acc), np.mean(f1), np.mean(iou)

def do_split(model,lst,exp,criterion,optimizer=None,global_plan=False, player_plan=False, incremental=False, device=DEVICE):
    data = []
    acc_loss = 0
    p = []
    g = []
    masks = []
    for batch, game in enumerate(lst):
        if exp==0:
            ground_truth = torch.tensor(game.global_plan_mat.reshape(-1)).float()
        elif exp==1:
            ground_truth = torch.tensor(game.partner_plan.reshape(-1)).float()
        elif exp==2:
            ground_truth = torch.tensor(game.global_diff_plan_mat.reshape(-1)).float()
            loss_mask = torch.tensor(game.global_plan_mat.reshape(-1)).float()
        else:
            ground_truth = torch.tensor(game.partner_diff_plan_mat.reshape(-1)).float()
            loss_mask = torch.tensor(game.plan_repr.reshape(-1)).float()
        
        prediction, _ = model(game, global_plan=global_plan, player_plan=player_plan, incremental=incremental)
        
        if incremental:
            ground_truth = ground_truth.to(device)
            g += [ground_truth for _ in prediction]
            masks += [loss_mask for _ in prediction]

            p += [x for x in prediction]

            data += list(zip(prediction.cpu().data.numpy(), [ground_truth.cpu().data.numpy()]*len(prediction),[batch]*len(prediction)))
        else:
            ground_truth = ground_truth.to(device)
            g.append(ground_truth)
            masks.append(loss_mask)

            p.append(prediction)
            
            data.append((prediction.cpu().data.numpy(), ground_truth.cpu().data.numpy(),batch))
        
        if (batch+1) % 2 == 0:
            loss = criterion(torch.stack(p),torch.stack(g), torch.stack(masks))
            
            loss += 1e-5 * sum(p.pow(2.0).sum() for p in model.parameters())
            if model.training and (not optimizer is None):
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), 1)
                # nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
            acc_loss += loss.item()
            p = []
            g = []
            masks = []
    
    acc_loss /= len(lst)

    acc0, acc, f1, iou = print_epoch(data,acc_loss,lst,exp)

    return acc0, acc_loss, data, acc, f1, iou


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def main(args):
    print(args, flush=True)
    print(f'PID: {os.getpid():6d}', flush=True)

    if isinstance(args.device, int) and args.device >= 0:
        DEVICE = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        print(f'Using {DEVICE}')
    else:
        print('Device must be a zero or positive integer, but got',args.device)
        exit()
    
    # if args.seed=='Random':
    #     pass
    # elif args.seed=='Fixed':
    #     random.seed(0)
    #     torch.manual_seed(1)
    #     np.random.seed(0)
    # else:
    #     print('Seed must be in [Random, Fixed], but got',args.seed)
    #     exit()

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
        
    if not args.intermediate in list(range(32)):
        print('Intermediate must be in',list(range(32)),', but got',args.intermediate)
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
        val    = [GameParser(f,d_flag,0,args.intermediate,d_move_flag) for f in dataset_splits['validation']]
        train  = [GameParser(f,d_flag,0,args.intermediate,d_move_flag) for f in dataset_splits['training']]
        if args.experiment > 2:
            val   += [GameParser(f,d_flag,4,args.intermediate,d_move_flag) for f in dataset_splits['validation']]
            train += [GameParser(f,d_flag,4,args.intermediate,d_move_flag) for f in dataset_splits['training']]
    elif args.pov=='Third':
        val    = [GameParser(f,d_flag,3,args.intermediate,d_move_flag) for f in dataset_splits['validation']]
        train  = [GameParser(f,d_flag,3,args.intermediate,d_move_flag) for f in dataset_splits['training']]
    elif args.pov=='First':
        val    = [GameParser(f,d_flag,1,args.intermediate,d_move_flag) for f in dataset_splits['validation']]
        train  = [GameParser(f,d_flag,1,args.intermediate,d_move_flag) for f in dataset_splits['training']]
        val   += [GameParser(f,d_flag,2,args.intermediate,d_move_flag) for f in dataset_splits['validation']]
        train += [GameParser(f,d_flag,2,args.intermediate,d_move_flag) for f in dataset_splits['training']]
    else:
        print('POV must be in [None, First, Third], but got', args.pov)
        exit()

    model = Model(seq_model,DEVICE).to(DEVICE)
    model.apply(init_weights)

    print(model)
    model.train()

    learning_rate = 1e-5
    weight_decay=1e-4

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    # optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    # optimizer = optim.Adadelta(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    criterion = PlanLoss()
    # criterion = torch.hub.load(
    #     'adeelh/pytorch-multi-class-focal-loss',
    #     model='focal_loss',
    #     alpha=[.25, .75],
    #     gamma=10,
    #     reduction='mean',
    #     device=device,
    #     dtype=torch.float32,
    #     force_reload=False
    # )
    # criterion = nn.BCEWithLogitsLoss(pos_weight=10*torch.ones(21*21).to(device))
    # criterion = nn.MSELoss()

    print(str(criterion), str(optimizer))

    num_epochs = 200#1#
    min_acc_loss = 1e6
    max_f1 = 0
    epochs_since_improvement = 0
    wait_epoch = 15#150#1000#
    max_fails = 5

    if args.model_path is not None:
        print(f'Loading {args.model_path}')
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
        acc, acc_loss, data, _, f1, iou = do_split(model,val,args.experiment,criterion,global_plan=global_plan, player_plan=player_plan, incremental=True, device=DEVICE)
        acc, acc_loss0, data, _, f1, iou = do_split(model,val,args.experiment,criterion,global_plan=global_plan, player_plan=player_plan, incremental=False, device=DEVICE)

        if np.mean([acc_loss,acc_loss0]) < min_acc_loss:
            min_acc_loss = np.mean([acc_loss,acc_loss0])
            epochs_since_improvement = 0
            print('^')
            torch.save(model.cpu().state_dict(), args.save_path)
            model = model.to(DEVICE)

        # data = list(zip(*data))
        # for x in data:
        #     a, b = list(zip(*x))
        # f1 = f1_score(a,b,average='weighted')
        # f1 = f1_score(a,b,average='weighted')
        # if (max_f1 < f1):
        #     max_f1 = f1
        #     epochs_since_improvement = 0
        #     print('^')
        #     torch.save(model.cpu().state_dict(), args.save_path)
        #     model = model.to(DEVICE)
    else:
        print('Training model from scratch', flush=True)

    for epoch in range(num_epochs):
        print(f'{os.getpid():6d} {epoch+1:4d},',end=' ',flush=True)
        shuffle(train)
        model.train()
        do_split(model,train,args.experiment,criterion,optimizer=optimizer,global_plan=global_plan, player_plan=player_plan, incremental=True, device=DEVICE)
        do_split(model,train,args.experiment,criterion,optimizer=optimizer,global_plan=global_plan, player_plan=player_plan, incremental=False, device=DEVICE)
        model.eval()
        acc, acc_loss, data, _, f1, iou = do_split(model,val,args.experiment,criterion,global_plan=global_plan, player_plan=player_plan, incremental=True, device=DEVICE)
        acc, acc_loss0, data, _, f1, iou = do_split(model,val,args.experiment,criterion,global_plan=global_plan, player_plan=player_plan, incremental=False, device=DEVICE)
        
        if np.mean([acc_loss,acc_loss0]) < min_acc_loss:
            min_acc_loss = np.mean([acc_loss,acc_loss0])
            epochs_since_improvement = 0
            print('^')
            torch.save(model.cpu().state_dict(), args.save_path)
            model = model.to(DEVICE)
        else:
            epochs_since_improvement += 1
            print()
        
        # test_val = iou
        # if (max_f1 < test_val):
        #     max_f1 = test_val
        #     epochs_since_improvement = 0
        #     print('^')
        #     if not args.save_path is None:
        #         torch.save(model.cpu().state_dict(), args.save_path)
        #     model = model.to(DEVICE)
        # else:
        #     epochs_since_improvement += 1
        #     print()

        if epoch > wait_epoch and epochs_since_improvement > max_fails:
            break
    print()
    print('Test')
    model.load_state_dict(torch.load(args.save_path))

    val = None
    train = None
    if args.pov=='None':
        test = [GameParser(f,d_flag,0,args.intermediate,d_move_flag) for f in dataset_splits['test']]
        if args.experiment > 2:
            test += [GameParser(f,d_flag,4,args.intermediate,d_move_flag) for f in dataset_splits['test']]
    elif args.pov=='Third':
        test = [GameParser(f,d_flag,3,args.intermediate,d_move_flag) for f in dataset_splits['test']]
    elif args.pov=='First':
        test  = [GameParser(f,d_flag,1,args.intermediate,d_move_flag) for f in dataset_splits['test']]
        test += [GameParser(f,d_flag,2,args.intermediate,d_move_flag) for f in dataset_splits['test']]
    else:
        print('POV must be in [None, First, Third], but got', args.pov)
    model.eval()
    acc, acc_loss, data, _, f1, iou = do_split(model,test,args.experiment,criterion,global_plan=global_plan, player_plan=player_plan, incremental=True, device=DEVICE)
    acc, acc_loss, data, _, f1, iou = do_split(model,test,args.experiment,criterion,global_plan=global_plan, player_plan=player_plan, incremental=False, device=DEVICE)
    
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
                    help='point of view [0:Global, 1:Partner, 2:GlobalDif, 3:PartnerDif]')
    parser.add_argument('--intermediate', type=int, 
                    help='point of view [0:Global, 1:Partner, 2:GlobalDif, 3:PartnerDif]')
    parser.add_argument('--save_path', type=str, 
                    help='path where to save model')
    parser.add_argument('--seed', type=int, 
                    help='Selet random seed by index [0, 1, 2, ...]. 0 -> random seed set to 0. n>0 -> random seed '
                    'set to n\'th random number with original seed set to 0')
    parser.add_argument('--device', type=int, default=0,
                    help='select cuda device number')
    parser.add_argument('--model_path', type=str, default=None,
                    help='path to the pretrained model to be loaded')
    
    main(parser.parse_args())