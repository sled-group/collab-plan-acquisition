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
from copy import deepcopy
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import entropy

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def onehot(x,n):
    retval = np.zeros(n)
    if x > 0:
        retval[x-1] = 1
    return retval

def print_epoch(data,lst,exp):

    # bins = [[]]

    # games = {}
    # for x in data:
    #     game_len = x[-1]
    #     frame_no = x[-2]//10
    #     while frame_no >= len(bins):
    #         bins.append([])
    #         # print(len(bins), frame_no)
    #     bins[frame_no].append(x)

    # out_bins = [[] for _ in range(len(bins))]


    num_bins = 1
    bins = [[] for _ in range(num_bins+1)]
    out_bins = [[] for _ in range(len(bins))]
    games = {}
    for x in data:
        game_len = x[-1]
        frame_no = x[-2]
        bin_size = game_len//num_bins + int(game_len%num_bins > 0)
        bin_idx = frame_no//bin_size + int(frame_no%bin_size>0)
        # print(x[2:], bin_size, bin_idx)
        bins[bin_idx].append(x)


    # for x in data:
    #     if not x[2] in games:
    #         games[x[2]] = []
    #     games[x[2]].append(x)
    # for k, g in games.items():
    #     bin_size = len(g)//num_bins + int(len(g)%num_bins > 0)
    #     for i, gg in enumerate(g):
    #         bins[i//bin_size + int(i%bin_size>0)].append(gg)

    # data = list(zip(*data))
    for bin_id in range(len(bins)):
        data_bin = bins[bin_id]
        acc = []
        prec = []
        rec = []
        f1 = []
        iou = []
        total = []
        predicts = []
        targets = []
        info_dict_lst = []
        stab_dict_lst = []
        # data += list(zip(
        #     prediction.cpu().data.numpy(),                        0
        #     [ground_truth.cpu().data.numpy()]*len(prediction),    1   
        #     [game_no]*len(prediction),                            2
        #     [prefix]*len(prediction),                             3
        #     [seed_num]*len(prediction),                           4
        #     list(range(len(prediction))),                         5
        #     [len(prediction)]*len(prediction)                     6
        #     ))
        for x in data_bin:
            game = lst[x[2]]
            seed = x[4]
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

            stab_dict_lst.append(json.loads('{' + x[3] + '}'))
            stab_dict_lst[-1]['Seed'] = seed
            stab_dict_lst[-1]['Bin']  = bin_id
            stab_dict_lst[-1]['Game'] = x[2]
            stab_dict_lst[-1]['V'] = deepcopy(a.reshape(21,21))
            # print(a)
            # exit()
            stab_dict_lst[-1]['L'] = deepcopy(b.reshape(21,21))

            # print()
            # print(a)
            # print(a,b)
            # exit()
            f1l = []
            b = b.reshape(21,21)
            if exp == 3:
                a = a.reshape(21,21)
                for idx,aa in enumerate(a):
                    if idx in possible_extra:
                        cand_idxs = set([i for i,x in enumerate(pov_plan_mat[idx]) if x])
                        
                        th, _ = zip(*sorted([(i, x) for i, x in enumerate(aa) if i in possible_mats], key=lambda x:x[1])[-2:])

                        temp_dict = json.loads('{' + x[3] + '}')
                        temp_dict['Seed'] = seed
                        temp_dict['Bin']  = bin_id
                        temp_dict['Game'] = x[2]
                        temp_dict['Mat']  = idx

                        # print(b[idx])
                        # exit()
                        if len(cand_idxs.intersection(set(th))):
                            temp_dict['V'] = abs(sum(pov_plan_mat[idx]*aa) - sum(b[idx]*aa))#pov_plan_mat[idx]*aa#deepcopy(aa)#
                            for jdx, _ in enumerate(aa):
                                a[idx,jdx] = pov_plan_mat[idx,jdx]
                            temp_dict['P'] = int(sum(pov_plan_mat[idx])//2)
                            temp_dict['L'] = int(sum(b[idx])//2)
                            # print(a[idx])
                            # print(b.reshape(21,21)[idx])
                            # print(b.reshape(21,21)[idx]-a[idx])
                            # print(1 if sum(abs((b.reshape(21,21)[idx]-a[idx]))) else 0)
                            # exit()
                        else:
                            temp_dict['V'] = abs(sum(pov_plan_mat[idx]*aa) - sum(b[idx]*aa)) #sum(pov_plan_mat[idx]*aa)#pov_plan_mat[idx]*aa#deepcopy(aa)#
                            for jdx, _ in enumerate(aa):
                                a[idx,jdx] = 0
                            
                            temp_dict['P'] = 0 #int(sum(pov_plan_mat[idx]))
                            temp_dict['L'] = int(sum(b[idx])//2)

                        info_dict_lst.append(temp_dict)
                        # print(temp_dict)
                        # exit()
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

            # b = b.reshape(21,21)
            # print()
            # print(a)
            # print(b)
            # print(b-a)
            # print(b+a)
            # exit()
            

            a = a.reshape(-1)
            b = b.reshape(-1)

            stab_dict_lst[-1]['P'] = a
            
            sa = set([i for i,x in enumerate(a) if x])
            sb = set([i for i,x in enumerate(b) if x])
            i = len(sa.intersection(sb))
            u = len(sa.union(sb))

            if u > 0:
                a,b = zip(*[(x,y) for x,y in zip(a,b) if x+y > 0])
                
            f1.append(f1_score(b,a,zero_division=1))
            f1l.append(f1_score(b,a,zero_division=1))
            prec.append(precision_score(b,a,zero_division=1))
            rec.append(recall_score(b,a,zero_division=1))
            # print(len(b),len(a))
            iou += [i/u if u > 0 else 1] * max(1,len(game.plan['full'])//2)
            total.append(sum(a))
            
            acc.append(iou[-1]>=0.5)
        out_bins[bin_id] = np.std(f1), np.mean(acc), np.mean(f1), np.mean(iou), predicts, targets, [x for x in iou], np.std(iou), np.mean(iou), info_dict_lst, stab_dict_lst
    # exit()
    retval = list(zip(*out_bins))
    return retval

    print('', end=' ',flush=True)
    targets = [t/2 for t in targets]*2
    predicts = [t/2 for t in predicts]*2
    return accuracy_score(targets,predicts), np.mean(acc), np.mean(f1), np.mean(iou), predicts, targets, [x for x in iou], f1_score(targets,predicts), np.mean(iou)

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

def do_split(model, load_str,lst,exp,baseline=None,baseline2=None,global_plan=False, player_plan=False, incremental=False, prefix=''):
    data = []
    acc_loss = 0
    p = []
    g = []
    for seed, load_path in enumerate(load_str):
        seed_num = int(load_path.split('_')[-1].split('.')[0])
        model.load_state_dict(torch.load(load_path))
        for game_no, game in enumerate(lst):
            
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


            # print(prediction.shape)
            
            # prediction = torch.stack([
            #    torch.sum(prediction[:idx],axis=0)/idx for idx in range(1,prediction.shape[0]+1)
            # ])

            # print(prediction.shape)
            # exit(0)

            data += list(zip(prediction.cpu().data.numpy(), [ground_truth.cpu().data.numpy()]*len(prediction),[game_no]*len(prediction),[prefix]*len(prediction),[seed_num]*len(prediction),list(range(len(prediction))),[len(prediction)]*len(prediction) ))

            # for idx in range(len(y)%10-1,len(y),10):
            #     for yy in y[idx:idx+1]:#[len(y)%10-1::10]:#
            #         prediction = model.plan_out(torch.cat((yy,torch.tensor(game.plan_repr.reshape(-1)).float().to(DEVICE))))
            #         prediction = F.softmax(prediction.reshape(21,21),-1).reshape(-1)
            #         # if exp == 3:
            #         #     prediction = torch.sum(prediction.reshape(21,21),dim=-1)/21
            #         # p.append(prediction)
                    
            #         data.append((prediction.cpu().data.numpy(), ground_truth.cpu().data.numpy()))
           
    
    list_str_fun = lambda lst: '['+' '.join([f'{x:5.3f}' for x in lst])+']'
    
    # print()
    f1_std, acc, f1, iou, predicts, targets, acc2, iou_std, iou_avg, info_dict_lst, stab_dict_lst = print_epoch(data,lst,exp)

    # print(stab_dict_lst[-1][-1])
    # print(info_dict_lst[-1][-1])


    entr_lst = []
    expected_lst = []
    stdev_lst = []
    entr_lst1 = []
    expected_lst1 = []
    stdev_lst1 = []
    entr_lst2 = []
    stdev_lst2 = []
    pred_lst = []
    val_lst = []
    lbl_lst = []
    for bin_lst in info_dict_lst[1:]:
        data = {}
        for x in bin_lst:
            # key = f"{x['Bin']}{x['Game']}"
            # key = f"{x['Seed']}{x['Bin']}"
            key = f"{x['Seed']}|{x['Game']}|{x['Mat']}"
            # key = f"{x['Seed']}|{x['Game']}"
            # key = f"{x['Seed']}{x['Bin']}{x['Game']}{x['Mat']}"
            if key not in data:
                data[key] = []
            # print(print(key),len(x['V']),len(x['P']),len(x['L']))
            data[key].append((x['V'],x['P'],x['L']))
        mi = []
        entr = []
        stdev = []
        expected = []
        entr1 = []
        stdev1 = []
        expected1 = []
        entr2 = []
        predictions = 0
        values = 0
        labels = 0
        # stdev2 = []
        for _, val in data.items():
            V, P, L = zip(*val)
            # print(V, P, L)
            # exit()

            # predictions += np.array(P)
            # values += np.array(V)
            # labels += np.array(L)

            change  = lambda x: [abs(x[i+1]-x[i]) for i in range(len(x)-1)]
            accumulation = lambda x: [sum(x[:i+1]) for i in range(len(x))]

            V = change(V)
            P = change(P)

            # V = accumulation(V)
            # P = accumulation(P)

            # entr.append(entropy(V,P))
            # entr.append(entropy(V))
            entr.append(accumulation(V))
            # expected.append(np.mean(V))
            # stdev.append(np.std(V))

            # entr1.append(entropy(V,L))
            # entr1.append(entropy(P))
            entr1.append(accumulation(P))
            # expected1.append(np.mean(P))
            # stdev1.append(np.std(P))

            # entr2.append(entropy(P,L))
            # stdev2.append(np.std(V))

            # if len(L) > 1:
            #     mi.append(mutual_info_regression(P,L,n_neighbors=1))
        # print(len(list(data.keys())), np.mean([len(v) for v in data.values()]), np.mean(entr), np.mean(stdev))
        # entr = [x for x in entr if not np.isnan(x) and not np.isinf(x)]
        # entr1 = [x for x in entr1 if not np.isnan(x) and not np.isinf(x)]
        # entr_lst.append(entr)
        entr_lst += entr
        # expected_lst.append(np.mean(expected))
        # stdev_lst.append(np.mean(stdev))
        # entr_lst1.append(entr1)
        entr_lst1 += entr1
        # expected_lst1.append(np.mean(expected1))
        # stdev_lst1.append(np.mean(stdev1))
        # entr_lst2.append(np.mean(entr2))
        # stdev_lst2.append(np.mean(stdev2))
        # pred_lst.append(predictions)
        # val_lst.append(values)
        # lbl_lst.append(labels)

    rez_lst = [[] for _ in range(max([len(x) for x in entr_lst]))]
    for xlst in entr_lst:
        for i,x in enumerate(xlst):
            rez_lst[i].append(x)
    rez_lst = [np.mean(x) for x in rez_lst]

    rez_lst1 = [[] for _ in range(max([len(x) for x in entr_lst1]))]
    for xlst in entr_lst1:
        for i,x in enumerate(xlst):
            rez_lst1[i].append(x)
    
    rez_lst1 = [np.mean(x) for x in rez_lst1]
    # print(len(entr_lst),len(entr_lst[0]),min([len(x) for x in entr_lst]),max([len(x) for x in entr_lst]),sorted([len(x) for x in entr_lst]))
    # exit()


    # entr_lst = [sum(x)/len(entr_lst) for x in zip(*entr_lst)]
    # entr_lst1 = [sum(x)/len(entr_lst1) for x in zip(*entr_lst1)]

    # stdev_mat_lst = []
    # mi_lst = []

    # for bin_lst in stab_dict_lst[1:]:
    #     data = {}
    #     for x in bin_lst:
    #         key = f"{x['Seed']}{x['Bin']}{x['Game']}"
    #         if key not in data:
    #             data[key] = []
    #         data[key].append((x['V'],x['P'],x['L']))
    #     stdev = []
    #     mi = []
    #     for _, val in data.items():
    #         V, P, L = zip(*val)
    #         m = sum(V)/len(V)
    #         # print(m.shape)
    #         m2 = sum([x*x for x in V])/len(V)
    #         # print(m2.shape)
    #         # print(m2-m*m)
    #         stdev.append(np.mean(np.sqrt(np.clip(m2-m*m,a_min=0,a_max=None))))
            
    #         # exit()
    #         for v, p, l in val:
    #             # mi.append(mutual_info_regression(v,np.sum(l,axis=-1),n_neighbors=1))
    #             mi.append(0)
    #     mi_lst.append(np.mean(mi))
    #     stdev_mat_lst.append(np.max(stdev))

    str_fun = lambda lst: '\n['+', '.join([f'{x:0.5f}' for x in lst]) +'] & '
    # str_fun = lambda lst: lst
    # print(str_fun(entr_lst), str_fun(stdev_lst), str_fun(mi_lst), str_fun(stdev_mat_lst),flush=True,end='')
    print(str_fun(rez_lst), str_fun(rez_lst1), flush=True,end='')
    # print(str_fun(entr_lst), str_fun(expected_lst), str_fun(stdev_lst), str_fun(entr_lst1), str_fun(stdev_lst1), str_fun(expected_lst1), str_fun(entr_lst2), str_fun(stdev_lst2), flush=True,end='')
    # print(pred_lst)
    # print(val_lst)
    # print(lbl_lst)

    # exit()
    # if not baseline is None:
    #     x1 = [int(x==y) for x, y in zip(baseline,targets)]
    #     x2 = [int(x==y) for x, y in zip(predicts,targets)]
    #     a = sum([1 for x, y in zip(x1,x2) if     x and     y])
    #     b = sum([1 for x, y in zip(x1,x2) if     x and not y])
    #     c = sum([1 for x, y in zip(x1,x2) if not x and     y])
    #     d = sum([1 for x, y in zip(x1,x2) if not x and not y])
    #     table = np.array([[a,b],[c,d]])
    #     # print(' '.join(f'{x:5.3f}' for x in mcnemar(table)), f'[[{a:2d}, {b:2d}], [{c:2d}, {d:2d}]]', end=' ')
    #     # print(f'{f1_edge:5.3f} & ','P $<$ ', ' '.join(f'{x:5.3f}' for x in mcnemar(table)[1:]), end=' &\t')
    #     print(f'{list_str_fun(f1),list_str_fun(f1_std)} & ', end=' &\t',flush=True)
    # else:
    #     print(f'{list_str_fun(f1),list_str_fun(f1_std)} & ','            ', end=' &\t',flush=True)
    #     # exit()
    # if not baseline2 is None:
    #     # x1 = baseline2
    #     # x2 = acc2
    #     # a = sum([1 for x, y in zip(x1,x2) if     x and     y])
    #     # b = sum([1 for x, y in zip(x1,x2) if     x and not y])
    #     # c = sum([1 for x, y in zip(x1,x2) if not x and     y])
    #     # d = sum([1 for x, y in zip(x1,x2) if not x and not y])
    #     # table = np.array([[a,b],[c,d]])
    #     # print(' '.join(f'{x:5.3f}' for x in mcnemar(table)), f'[[{a:2d}, {b:2d}], [{c:2d}, {d:2d}]]', end=' ')

    #     # print(f'{f1_avg:5.3f} ({np.std(acc2):5.3f}) ({np.mean([a-b for a, b in zip(acc2,baseline2)]):5.3f},{np.std([a-b for a, b in zip(acc2,baseline2)]):5.3f}) & ','P $<$ ', f'{ttest_rel(baseline2,acc2).pvalue:5.3f}',end=' ')
    #     # print(f'{f1_avg:5.3f} ({np.std(acc2):5.3f}) ({np.mean([int(a>b) for a, b in zip(acc2,baseline2)]):5.3f}) & ','P $<$ ', f'{ttest_rel(baseline2,acc2).pvalue:5.3f}',end=' ')
    #     print(f'{list_str_fun(iou_avg),list_str_fun(iou_std)} & ',end=' ',flush=True)
        
    #     # x1 = [int(x>=0.5) for x in baseline2]
    #     # x2 = [int(x>=0.5) for x in acc2]
    #     # a = sum([1 for x, y in zip(x1,x2) if     x and     y])
    #     # b = sum([1 for x, y in zip(x1,x2) if     x and not y])
    #     # c = sum([1 for x, y in zip(x1,x2) if not x and     y])
    #     # d = sum([1 for x, y in zip(x1,x2) if not x and not y])
    #     # table = np.array([[a,b],[c,d]])
    #     # print(f'{f1_avg:5.3f} & ','P $<$ ', ' '.join(f'{x:5.3f}' for x in mcnemar(table)[1:]),end=' ')
        
    #     # print(list(zip(acc2,baseline)))
    # else:
    #     print(f'{list_str_fun(iou_avg),list_str_fun(iou_std)}  & ',f'            ',end=' ',flush=True)
    #     # exit()
    # print('\\\\',end=' ')
    
    
    # exit()
    return f1_std, acc_loss, data, acc, f1, iou, predicts, targets, acc2, data


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

    # # print('Test')
    # args.intermediate = 0
    # load_str = sorted(glob(f'{args.save_path.split("_seed_")[0]}_seed_*.torch'))
    # # print(load_str)
    # # exit(0)
    # # load_str = 'models/plan_no_dlg/plan_exp3_LSTM_int3_seed_0.torch'
    # # load_str = 'models/plan_no_dlg/plan_exp3_LSTM_int0_seed_0.torch'
    # # load_str = 'models/plan_vid_only/plan_exp3_LSTM_int0_seed_0.torch'
    # # print(load_str,end='\t')
    # for _ in range(4):
    #     print('  & ',end='\t')
    # # model.load_state_dict(torch.load(load_str))
    
    # # model.load_state_dict(torch.load(args.save_path))

    # val = None
    # train = None
    # if args.pov=='None':
    #     test = [GameParser(f,d_flag,0,args.intermediate,d_move_flag) for f in dataset_splits['test']]
    #     if args.experiment > 2:
    #         test += [GameParser(f,d_flag,4,args.intermediate,d_move_flag) for f in dataset_splits['test']]
    # elif args.pov=='Third':
    #     test = [GameParser(f,d_flag,3,args.intermediate,d_move_flag) for f in dataset_splits['test']]
    # elif args.pov=='First':
    #     test  = [GameParser(f,d_flag,1,args.intermediate,d_move_flag) for f in dataset_splits['test']]
    #     test += [GameParser(f,d_flag,2,args.intermediate,d_move_flag) for f in dataset_splits['test']]
    # else:
    #     print('POV must be in [None, First, Third], but got', args.pov)
    # model.eval()
    # acc0, acc_loss, data, acc, f1, iou, predicts, targets, acc2, data = do_split(model, load_str,test,args.experiment,global_plan=global_plan, player_plan=player_plan)
    # print()
    
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
                acc0, acc_loss, data, acc, f1, iou, predicts2, targets, _, data = do_split(model,load_str,test,args.experiment,global_plan=global_plan, player_plan=player_plan, prefix=f'"Dlg": {int(d_flag)}, "Move": {int(d_move_flag)}, "Interm": {args.intermediate}')
                print()
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