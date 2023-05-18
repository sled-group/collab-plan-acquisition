from glob import glob
import os, json, sys
import torch, random, torch.nn as nn, numpy as np
from torch import optim
from random import shuffle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.data.game_parser import GameParser, make_splits, set_seed
from src.models.plan_model import Model
from src.models.losses import PlanLoss
import argparse
from mlxtend.evaluate import mcnemar
from torch.nn import functional as F
from scipy.stats import ttest_rel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def onehot(x,n):
    retval = np.zeros(n)
    if x > 0:
        retval[x-1] = 1
    return retval

def print_epoch(data,lst,exp):
    # data = list(zip(*data))
    acc = []
    prec = []
    rec = []
    f1 = []
    iou = []
    total = []
    predicts = []
    targets = []
    for x in data:
        game = lst[x[2]]
        game_mats = game.plan['materials']
        pov_plan = game.plan[f'player{game.pov}']
        pov_plan_mat = game.__dict__[f'player{game.pov}_plan_mat']
        possible_mats = [game.materials_dict[x]-1 for x,_ in zip(game_mats[1:],pov_plan[1:])]
        possible_cand = [game.materials_dict[x]-1 for x,y in zip(game_mats[1:],pov_plan[1:]) if y['make'] and y['make'][0][0]==-1]
        possible_extra = [game.materials_dict[x]-1 for x,y in zip(game_mats[1:],pov_plan[1:]) if y['make'] and y['make'][0][0]>-1]
        # print(possible_mats, possible_cand)
        # print(possible_mats)
        # exit()
        a, b = x[:2]
        # print()
        # print(a)
        # print(a,b)
        # exit()
        if exp == 3:
            a = a.reshape(21,21)
            for idx,aa in enumerate(a):
                if idx in possible_extra:
                    cand_idxs = set([i for i,x in enumerate(pov_plan_mat[idx]) if x])
                    th, _ = zip(*sorted([(i, x) for i, x in enumerate(aa) if i in possible_mats], key=lambda x:x[1])[-2:])
                    # print(th)
                    # print(cand_idxs)
                    # print(cand_idxs.intersection(set(th)))
                    # exit()
                    if len(cand_idxs.intersection(set(th))):
                        for jdx, _ in enumerate(aa):
                            a[idx,jdx] = pov_plan_mat[idx,jdx]
                        # print(a[idx])
                        # print(b.reshape(21,21)[idx])
                        # print(b.reshape(21,21)[idx]-a[idx])
                        # print(1 if sum(abs((b.reshape(21,21)[idx]-a[idx]))) else 0)
                        # exit()
                    else:
                        for jdx, _ in enumerate(aa):
                            a[idx,jdx] = 0
                    if sum(abs((b.reshape(21,21)[idx]-a[idx]))):
                        predicts.append(2-sum(b.reshape(21,21)[idx]))
                        targets.append(sum(b.reshape(21,21)[idx]))
                    else:                        
                        predicts.append(sum(b.reshape(21,21)[idx]))
                        targets.append(sum(b.reshape(21,21)[idx]))
                    # print(a[idx])
                    # print(b.reshape(21,21)[idx])
                    # print(abs(b.reshape(21,21)[idx]-a[idx]),sum(abs(b.reshape(21,21)[idx]-a[idx])))
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
        # predicts.append(np.argmax(a))
        # targets.append(np.argmax(a) if np.argmax(a) in [x for x in b if x] else np.argmax(b))
        # ma = np.mean(a*b)#np.mean([x*y for x,y in zip(a,b)])
        # mb = np.mean(a*(1-b))#np.mean([x*(1-y) for x,y in zip(a,b)])
        # th = np.mean([ma]+[mb])
        # th = np.mean([max(a)]*9+[min(a)])
        # # print(f'({ma:0.2f},{th:0.2f},{mb:0.2f})',end='')
        # # th = 0.5
        # a = [x >= th for x in a] #if ma > mb else [x <= th for x in a]
        # print((a-b).reshape(21,21))
        # acc.append(accuracy_score(a,b))
        # f1.append(f1_score(a,b,average="weighted"))
        sa = set([i for i,x in enumerate(a) if x])
        sb = set([i for i,x in enumerate(b) if x])
        i = len(sa.intersection(sb))
        u = len(sa.union(sb))
        
        # print(sa,sb,i,u,i/u if u > 0 else 1)
        # i = sum([int(x+y == 2) for x,y in zip(a,b)])
        # u = sum([int(x+y >  0) for x,y in zip(a,b)])
        # f1.append(f1_score(a,b,average='weighted',zero_division=1))
        # prec.append(precision_score(a,b,average='weighted',zero_division=1))
        # rec.append(recall_score(a,b,average='weighted',zero_division=1))
        if u > 0:
            a,b = zip(*[(x,y) for x,y in zip(a,b) if x+y > 0])
            
        f1.append(f1_score(b,a,zero_division=1))
        # f1.append(accuracy_score(b,a))
        prec.append(precision_score(b,a,zero_division=1))
        rec.append(recall_score(b,a,zero_division=1))
        # print(len(b),len(a))
        iou += [i/u if u > 0 else 1] * max(1,len(game.plan['full'])//2)
        total.append(sum(a))
        
        acc.append(iou[-1]>=0.5)
    # exit()
    # print(f'({np.min(acc):5.3f},{np.min(f1):5.3f},{np.min(iou):5.3f})', end=' ',flush=True)
    # print(targets)
    # print(predicts)
    # print(len(predicts))
    # print(
    #       f'({accuracy_score(targets,predicts):5.3f},'
    #       f'{precision_score(targets,predicts,average="weighted"):5.3f},'
    #       f'{recall_score(targets,predicts,average="weighted"):5.3f},'
    #       f'{f1_score(targets,predicts,average="weighted"):5.3f}'
    #        '),('
    #       f'{np.mean(acc):5.3f},'
    #       f'{np.mean(prec):5.3f},'
    #       f'{np.mean(rec):5.3f},'
    #       f'{np.mean(f1):5.3f},'
    #       f'{np.mean(iou):5.3f},'
    #       f'{np.std(iou):5.3f},'
    #       f'{np.mean(total):5.3f})'
    #     , len(predicts)
    #     ,
    #       end=' ',flush=True)
    # print(
    #       f'{f1_score(targets,predicts,average="weighted"):5.3f}'
    #       ' &\t'
    #       f'{np.mean(f1):5.3f}'
    #       ' &\t'
    #     ,
    #       end=' ',flush=True)
    # print(f'({np.max(acc):5.3f},{np.max(f1):5.3f},{np.max(iou):5.3f})', end=' ',flush=True)
    # exit(0)
        # if max(a) <= 1:
        #     print(f'({accuracy_score(a,b):5.3f},{f1_score(a,b,average="weighted"):5.3f},{sum(a)/len(a):5.3f},{sum(b)/len(b):5.3f},{len(b)})', end=' ',flush=True)
        # else:
        #     print(f'({accuracy_score(a,b):5.3f},{f1_score(a,b,average="weighted"):5.3f},{len(b)})', end=' ',flush=True)
    print('', end=' ',flush=True)
    targets = [t/2 for t in targets]*2
    predicts = [t/2 for t in predicts]*2
    # print(len(targets))
    # print(targets)
    # print(predicts)
    # return accuracy_score(targets,predicts), np.mean(acc), np.mean(f1), np.mean(iou), predicts, targets, f1, f1_score(targets,predicts), np.mean(f1)
    # return accuracy_score(targets,predicts), np.mean(acc), np.mean(f1), np.mean(iou), predicts, targets, [x for x in iou], f1_score(targets,predicts), np.mean(iou)
    return accuracy_score(targets,predicts), np.mean(acc), np.mean(f1), np.mean(iou), predicts, targets, [x for x in iou], accuracy_score(targets,predicts), np.mean(iou)
    # return accuracy_score(targets,predicts), np.mean(acc), np.mean(f1), np.mean(iou), predicts, targets, f1, f1_score(targets,predicts,average="weighted"), np.mean(f1)

def make_splits(split_file = 'config/dataset_splits.json'):
    if not os.path.isfile(split_file):
        dirs = sorted(glob('data/*_logs/*'))
        games = sorted(list(map(GameParser, dirs)), key=lambda x: len(x.question_pairs), reverse=True)

        test = games[0::5]
        val = games[1::5]
        train = games[2::5]+games[3::5]+games[4::5]

        dataset_splits = {'test' : [g.game_path for g in test], 'validation' : [g.game_path for g in val], 'training' : [g.game_path for g in train]}
        json.dump(dataset_splits, open(split_file,'w'))
    else:
        dataset_splits = json.load(open(split_file))
    
    return dataset_splits

def do_split(model, load_str,lst,exp,baseline=None,baseline2=None,global_plan=False, player_plan=False, incremental=False):
    data = []
    acc_loss = 0
    p = []
    g = []
    for load_path in load_str:
        model.load_state_dict(torch.load(load_path))
        for batch, game in enumerate(lst):
            
            # print(exp)
            # print(game.global_plan_mat.reshape(-1).shape)
            # print(game.player1_plan_mat.reshape(-1).shape)
            # print(game.player2_plan_mat.reshape(-1).shape)
            # print(game.load_player1)
            # print(game.load_player2)
            # print(prediction.shape)
            # print()
            # print(ground_truth)

            if exp==0:
                ground_truth = torch.tensor(game.global_plan_mat.reshape(-1)).float()
            elif exp==1:
                ground_truth = torch.tensor(game.partner_plan.reshape(-1)).float()
            elif exp==2:
                ground_truth = torch.tensor(game.global_diff_plan_mat.reshape(-1)).float()
                # ground_truth = torch.clamp(torch.sum(ground_truth.reshape(21,21),dim=-2),0,1)
            else:
                ground_truth = torch.tensor(game.partner_diff_plan_mat.reshape(-1)).float()
                # ground_truth = torch.clamp(torch.sum(ground_truth.reshape(21,21),dim=-1),0,1)

            # ground_truth = torch.tensor(game.plan_repr.reshape(-1)).float()
            # print()
            # print(ground_truth)
            # exit()
            # ground_truth = torch.clamp(torch.sum(ground_truth.reshape(21,21),dim=-1),0,1)
            
            # print(ground_truth)

            ground_truth = ground_truth.to(DEVICE)
            # g.append(ground_truth)
            
            prediction, y = model(game, global_plan=global_plan, player_plan=player_plan, incremental=True)

            data += list(zip(prediction.cpu().data.numpy(), [ground_truth.cpu().data.numpy()]*len(prediction),[batch]*len(prediction)))

            # for idx in range(len(y)%10-1,len(y),10):
            #     for yy in y[idx:idx+1]:#[len(y)%10-1::10]:#
            #         prediction = model.plan_out(torch.cat((yy,torch.tensor(game.plan_repr.reshape(-1)).float().to(DEVICE))))
            #         prediction = F.softmax(prediction.reshape(21,21),-1).reshape(-1)
            #         # if exp == 3:
            #         #     prediction = torch.sum(prediction.reshape(21,21),dim=-1)/21
            #         # p.append(prediction)
                    
            #         data.append((prediction.cpu().data.numpy(), ground_truth.cpu().data.numpy()))
           
    

    
    acc0, acc, f1, iou, predicts, targets, acc2, f1_edge, f1_avg = print_epoch(data,lst,exp)
    if not baseline is None:
        x1 = [int(x==y) for x, y in zip(baseline,targets)]
        x2 = [int(x==y) for x, y in zip(predicts,targets)]
        a = sum([1 for x, y in zip(x1,x2) if     x and     y])
        b = sum([1 for x, y in zip(x1,x2) if     x and not y])
        c = sum([1 for x, y in zip(x1,x2) if not x and     y])
        d = sum([1 for x, y in zip(x1,x2) if not x and not y])
        table = np.array([[a,b],[c,d]])
        # print(' '.join(f'{x:5.3f}' for x in mcnemar(table)), f'[[{a:2d}, {b:2d}], [{c:2d}, {d:2d}]]', end=' ')
        # print(f'{f1_edge:5.3f} & ','P $<$ ', ' '.join(f'{x:5.3f}' for x in mcnemar(table)[1:]), end=' &\t')
        print(f'{f1:5.3f} & ','P $<$ ', ' '.join(f'{x:5.3f}' for x in mcnemar(table)[1:]), end=' &\t')
    else:
        print(f'{f1_edge:5.3f} & ','            ', end=' &\t')
        # exit()
    if not baseline2 is None:
        # x1 = baseline2
        # x2 = acc2
        # a = sum([1 for x, y in zip(x1,x2) if     x and     y])
        # b = sum([1 for x, y in zip(x1,x2) if     x and not y])
        # c = sum([1 for x, y in zip(x1,x2) if not x and     y])
        # d = sum([1 for x, y in zip(x1,x2) if not x and not y])
        # table = np.array([[a,b],[c,d]])
        # print(' '.join(f'{x:5.3f}' for x in mcnemar(table)), f'[[{a:2d}, {b:2d}], [{c:2d}, {d:2d}]]', end=' ')

        # print(f'{f1_avg:5.3f} ({np.std(acc2):5.3f}) ({np.mean([a-b for a, b in zip(acc2,baseline2)]):5.3f},{np.std([a-b for a, b in zip(acc2,baseline2)]):5.3f}) & ','P $<$ ', f'{ttest_rel(baseline2,acc2).pvalue:5.3f}',end=' ')
        # print(f'{f1_avg:5.3f} ({np.std(acc2):5.3f}) ({np.mean([int(a>b) for a, b in zip(acc2,baseline2)]):5.3f}) & ','P $<$ ', f'{ttest_rel(baseline2,acc2).pvalue:5.3f}',end=' ')
        # print(f'{f1_avg:5.3f} & ','P $<$ ', f'{ttest_rel(baseline2,acc2).pvalue:5.3f}',end=' ')
        print(f'{f1_avg:5.3f} & ','P $<$ ', 0,end=' ')
        
        # x1 = [int(x>=0.5) for x in baseline2]
        # x2 = [int(x>=0.5) for x in acc2]
        # a = sum([1 for x, y in zip(x1,x2) if     x and     y])
        # b = sum([1 for x, y in zip(x1,x2) if     x and not y])
        # c = sum([1 for x, y in zip(x1,x2) if not x and     y])
        # d = sum([1 for x, y in zip(x1,x2) if not x and not y])
        # table = np.array([[a,b],[c,d]])
        # print(f'{f1_avg:5.3f} & ','P $<$ ', ' '.join(f'{x:5.3f}' for x in mcnemar(table)[1:]),end=' ')
        
        # print(list(zip(acc2,baseline)))
    else:
        print(f'{f1_avg:5.3f} ({np.std(acc2):5.3f}) & ',f'            ',end=' ')
        # exit()
    print('\\\\',end=' ')
    
    
    # exit()
    return acc0, acc_loss, data, acc, f1, iou, predicts, targets, acc2, data


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def main(args):
    # print(args)
    # print(f'PID: {os.getpid():6d}')
    
    # if isinstance(args.seed, int) and args.seed >= 0:
    #     seed = set_seed(args.seed)
    # else:
    #     print('Seed must be a zero or positive integer, but got',args.seed)
    #     exit()
    
    # dataset_splits = make_splits('config/dataset_splits.json')
    # dataset_splits = make_splits('config/dataset_splits_dev.json')
    # dataset_splits = make_splits('config/dataset_splits_old.json')
    dataset_splits = make_splits('config/dataset_splits_new.json')
    
    if args.use_dialogue_moves=='Yes':
        d_move_flag = True
    elif args.use_dialogue_moves=='No':
        d_move_flag = False
    else:
        print('Use dialogue must be in [Yes, No], but got',args.use_dialogue)
        exit()
    
    if args.use_dialogue=='Yes':
        d_flag = True
    elif args.use_dialogue=='No':
        d_flag = False
    else:
        print('Use dialogue must be in [Yes, No], but got',args.use_dialogue)
        exit()
        
    if not args.experiment in list(range(9)):
        print('Experiment must be in',list(range(9)),', but got',args.experiment)
        exit()
        
    if not args.intermediate in list(range(16)):
        print('Intermediate must be in',list(range(16)),', but got',args.intermediate)
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
    # print('global_plan', global_plan, 'player_plan', player_plan)

    model = Model(seq_model).to(DEVICE)
    # model.apply(init_weights)

    # print(model)
    # model.train()

    learning_rate = 1e-5
    num_epochs = 1000#1#
    weight_decay=1e-4

    # print('Test')
    args.intermediate = 0
    load_str = sorted(glob(f'{args.save_path.split("_seed_")[0]}_seed_*.torch'))
    # print(load_str)
    # exit(0)
    # load_str = 'models/plan_no_dlg/plan_exp3_LSTM_int3_seed_0.torch'
    # load_str = 'models/plan_no_dlg/plan_exp3_LSTM_int0_seed_0.torch'
    # load_str = 'models/plan_vid_only/plan_exp3_LSTM_int0_seed_0.torch'
    # print(load_str,end='\t')
    for _ in range(4):
        print('  & ',end='\t')
    # model.load_state_dict(torch.load(load_str))
    
    # model.load_state_dict(torch.load(args.save_path))

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
    acc0, acc_loss, data, acc, f1, iou, predicts, targets, acc2, data = do_split(model, load_str,test,args.experiment,global_plan=global_plan, player_plan=player_plan)
    print()
    
    for d_flag in [False, True]:
        for d_move_flag in [False, True]:
            for args.intermediate in range(0,8):
                load_str = sorted(glob(f'{args.root_path}/plan_exp3_{args.seq_model}_dlg_{"Yes" if d_flag else "No"}_move_{"Yes" if d_move_flag else "No"}_int{args.intermediate}_seed_*.torch'))

                # if args.intermediate < 16:
                #     load_str = f'{args.save_path.split("_seed_")[0][:-1]}{args.intermediate}_seed_0.torch'
                #     # print(load_str,end='\t')
                #     model.load_state_dict(torch.load(load_str))
                # else:
                #     load_str = f'models/plan_no_dlg/plan_exp3_LSTM_int{args.intermediate}_seed_0.torch'
                #     # print(load_str,end='\t')
                #     model.load_state_dict(torch.load(load_str))
                asdf = args.intermediate
                for _ in range(4):
                    if asdf % 2:
                        print('X & ',end='\t')
                    else: 
                        print('  & ',end='\t')
                    asdf = asdf//2
                
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
                # acc0, acc_loss, data, acc, f1, iou, predicts2, targets, _, data = do_split(model,load_str,test,args.experiment,baseline=predicts,baseline2=acc2,global_plan=global_plan, player_plan=player_plan)
                acc0, acc_loss, data, acc, f1, iou, predicts2, targets, _, data = do_split(model,load_str,test,args.experiment,global_plan=global_plan, player_plan=player_plan)
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
                    help='path to the baseline model')
    parser.add_argument('--root_path', type=str, 
                    help='directory to comparrisson models')
    
    main(parser.parse_args())