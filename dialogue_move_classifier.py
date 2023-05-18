from glob import glob
import os, json, sys
import torch, random, torch.nn as nn, numpy as np
from torch import optim
from random import shuffle
from sklearn.metrics import accuracy_score, f1_score
from src.data.game_parser import GameParser, make_splits, onehot, DEVICE, set_seed
from src.models.dialogue_move_classification_model import Model
from src.models.losses import DialogueMoveLoss
import argparse

def print_epoch(data,acc_loss,lst):
    print(f'{acc_loss/len(lst):9.4f}',end='; ',flush=True)
    # print(data[0])
    gt, pr = list(zip(*data))
    gtmv, gts1, gts2, gts3 = list(zip(*gt))
    prmv, prs1, prs2, prs3 = list(zip(*pr))


    ret_acc = []
    ret_f1 = []

    # gts1,prs1 = [(g,p) for g, p in zip(gts1,prs1) if g]
    # gts2,prs2 = [(g,p) for g, p in zip(gts2,prs2) if g]
    # gts3,prs3 = [(g,p) for g, p in zip(gts3,prs3) if g]

    # if sum(gts1):
    #     prs1, gts1 = zip(*[(a,b) for a,b in zip(prs1,gts1) if b])
    #     prs1 = torch.stack(prs1)
    #     gts1 = torch.stack(gts1)
    # if sum(gts2):
    #     prs2, gts2 = zip(*[(a,b) for a,b in zip(prs2,gts2) if b])
    #     prs2 = torch.stack(prs2)
    #     gts2 = torch.stack(gts2)
    # if sum(gts3):
    #     prs3, gts3 = zip(*[(a,b) for a,b in zip(prs3,gts3) if b])
    #     prs3 = torch.stack(prs3)
    #     gts3 = torch.stack(gts3)
    
    for x in [(gtmv, prmv), (gts1,prs1), (gts2,prs2), (gts3,prs3)]:
        # print(x)
        a, b = x #list(zip(*x))
        # print(a,b)
        if max(a) <= 1:
            ret_acc.append(accuracy_score(a,b))
            ret_f1.append(f1_score(a,b,average="weighted"))
            print(f'({accuracy_score(a,b):5.3f},{f1_score(a,b,average="weighted"):5.3f},{sum(a)/len(a):5.3f},{sum(b)/len(b):5.3f},{len(b)})', end=' ',flush=True)
        else:
            ret_acc.append(accuracy_score(a,b))
            ret_f1.append(f1_score(a,b,average="weighted"))
            print(f'({accuracy_score(a,b):5.3f},{f1_score(a,b,average="weighted"):5.3f})', end=' ',flush=True)
    print(len(b), end=' ',flush=True)
    print('', end='; ',flush=True)
    return ret_acc, ret_f1

def do_split(model,lst,exp,criterion,optimizer=None,global_plan=False, player_plan=False):
    data = []
    acc_loss = 0
    for game in lst:
        l = model(game, global_plan=global_plan, player_plan=player_plan)
        prediction = []
        ground_truth = []
        for gt, prd in l:
            
            # for idx,gtx in enumerate(gt[0][-1][1:]):
            #     if not gtx:
            #         # print(gt[0][-1], prd)
            #         prd[idx+1] *= 0
            #         prd[idx+1][0] = 1
            #         # print(gt[0][-1], prd)
            #         # exit()

                    
            
            prediction.append(prd)
            ground_truth.append(gt[0][-1])
            data.append((gt[0][-1],[torch.argmax(p.cpu()).item() for p in prd]))
            # print(data[-1])
            # exit()
            
        if prediction:
            prediction = [torch.stack(p) for p in zip(*prediction)]
        else:
            continue
        if ground_truth:
            # ground_truth = torch.stack(ground_truth).float().to(DEVICE)
            ground_truth = list(zip(*ground_truth))
            ground_truth = torch.tensor(ground_truth).long().to(DEVICE)
        else:
            continue
            
            
        loss  = criterion(prediction,ground_truth)
        loss += 1e-5 * sum(p.pow(2.0).sum() for p in model.parameters())
        # loss += 1e-5 * sum(p.abs().sum() for p in model.parameters())
        
        if model.training and (not optimizer is None):
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        acc_loss += loss.item()
    ret_acc, ret_f1 = print_epoch(data,acc_loss,lst)
    return acc_loss, data, ret_acc, ret_f1

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def main(args):
    print(args)
    print(f'PID: {os.getpid():6d}')

    if isinstance(args.seed, int) and args.seed >= 0:
        seed = set_seed(args.seed)
    else:
        print('Seed must be a zero or positive integer, but got',args.seed)
        exit()
    
    dataset_splits = make_splits()
    # dataset_splits['validation'] = dataset_splits['validation'][:2]
    # dataset_splits['training'] = dataset_splits['training'][:2]
    # dataset_splits['test'] = dataset_splits['test'][:2]
    
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
        

    if args.seq_model=='GRU':
        seq_model = 0
    elif args.seq_model=='LSTM':
        seq_model = 1
    elif args.seq_model=='Transformer':
        seq_model = 2
    elif args.seq_model=='None':
        seq_model = 3
    else:
        print('The sequence model must be in [GRU, LSTM, Transformer, None], but got', args.seq_model)
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
        val    = [GameParser(f,d_flag,0,0,True) for f in dataset_splits['validation']]
        train  = [GameParser(f,d_flag,0,0,True) for f in dataset_splits['training']]
        if args.experiment > 2:
            val   += [GameParser(f,d_flag,4,0,True) for f in dataset_splits['validation']]
            train += [GameParser(f,d_flag,4,0,True) for f in dataset_splits['training']]
    elif args.pov=='Third':
        val    = [GameParser(f,d_flag,3,0,True) for f in dataset_splits['validation']]
        train  = [GameParser(f,d_flag,3,0,True) for f in dataset_splits['training']]
    elif args.pov=='First':
        val    = [GameParser(f,d_flag,1,0,True) for f in dataset_splits['validation']]
        train  = [GameParser(f,d_flag,1,0,True) for f in dataset_splits['training']]
        val   += [GameParser(f,d_flag,2,0,True) for f in dataset_splits['validation']]
        train += [GameParser(f,d_flag,2,0,True) for f in dataset_splits['training']]
    else:
        print('POV must be in [None, First, Third], but got', args.pov)
        exit()

    model = Model(seq_model).to(DEVICE)
    model.apply(init_weights)

    print(model)
    model.train()

    learning_rate = 1e-6
    num_epochs = 1000#1#

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    # optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    # optimizer = optim.Adadelta(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    # criterion = nn.CrossEntropyLoss()
    criterion = DialogueMoveLoss(DEVICE)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([289,51,45,57,14,12,1,113,6,264,27,63,22,66,2,761,129,163,5]).to(DEVICE)/2090)
    # criterion = nn.MSELoss()

    print(str(criterion))
    print(str(optimizer))

    min_acc_loss = 100
    max_f1 = 0
    min_loss = 1e100
    epochs_since_improvement = 0
    wait_epoch = 50#100#
    max_wait_epochs = 20

    

    for epoch in range(num_epochs):
        print(f'{os.getpid():6d} {epoch+1:4d},',end=' ',flush=True)
        shuffle(train)
        model.train()
        do_split(model,train,args.experiment,criterion,optimizer=optimizer,global_plan=global_plan, player_plan=player_plan)
        model.eval()
        acc_loss, data, ret_acc, ret_f1 = do_split(model,val,args.experiment,criterion,global_plan=global_plan, player_plan=player_plan)

        # gt, pr = list(zip(*data))
        # gtmv, gts1, gts2, gts3 = list(zip(*gt))
        # prmv, prs1, prs2, prs3 = list(zip(*pr))
        
        # f1 =[]
        # for x in [(gtmv, prmv), (gts1,prs1), (gts2,prs2), (gts3,prs3)]:
        #     # print(x)
        #     a, b = x #list(zip(*x))
        #     f1.append(f1_score(a,b,average='weighted'))

        # f1 = np.mean(f1)
        if (max_f1 < np.mean(ret_f1[:1])):
            max_f1 = np.mean(ret_f1[:1])
        # if min_loss > acc_loss:
        #     min_loss = acc_loss
            epochs_since_improvement = 0
            print('^',flush=True)
            if not args.save_path is None:
                torch.save(model.cpu().state_dict(), args.save_path)
            model = model.to(DEVICE)
        else:
            epochs_since_improvement += 1
            print(flush=True)
            
        if epoch > wait_epoch and epochs_since_improvement > max_wait_epochs:
            break
    print(flush=True)
    print('Test',flush=True)
    model.load_state_dict(torch.load(args.save_path))

    val = None
    train = None
    if args.pov=='None':
        test = [GameParser(f,d_flag,0,0,True) for f in dataset_splits['test']]
        if args.experiment > 2:
            test += [GameParser(f,d_flag,4,0,True) for f in dataset_splits['test']]
    elif args.pov=='Third':
        test = [GameParser(f,d_flag,3,0,True) for f in dataset_splits['test']]
    elif args.pov=='First':
        test  = [GameParser(f,d_flag,1,0,True) for f in dataset_splits['test']]
        test += [GameParser(f,d_flag,2,0,True) for f in dataset_splits['test']]
    else:
        print('POV must be in [None, First, Third], but got', args.pov,flush=True)
    model.eval()
    _, data, ret_acc, ret_f1 = do_split(model,test,args.experiment,criterion,global_plan=global_plan, player_plan=player_plan)
    
    print()
    print(data,flush=True)
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
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
    parser.add_argument('--seed', type=int, 
                    help='Use random or fixed seed [Random, Fixed]')
    
    main(parser.parse_args())