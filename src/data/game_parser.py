from email.mime import base
from glob import glob
import os, string, json, pickle
import torch, random, numpy as np
from transformers import BertTokenizer, BertModel
import cv2
import imageio
from src.data.action_extractor import proc_action


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed_idx):
    seed = 0
    random.seed(0)
    for _ in range(seed_idx):
        seed = random.random()
    random.seed(seed)
    torch.manual_seed(seed)
    print('Random seed set to', seed)
    return seed


def make_splits(split_file = 'config/dataset_splits.json'):
    if not os.path.isfile(split_file):
        dirs = sorted(glob('data/saved_logs/*') + glob('data/main_logs/*'))
        games = sorted(list(map(GameParser, dirs)), key=lambda x: len(x.question_pairs), reverse=True)

        test = games[0::5]
        val = games[1::5]
        train = games[2::5]+games[3::5]+games[4::5]

        dataset_splits = {'test' : [g.game_path for g in test], 'validation' : [g.game_path for g in val], 'training' : [g.game_path for g in train]}
        json.dump(dataset_splits, open('config/dataset_splits_old.json','w'), indent=4)


        dirs = sorted(glob('data/new_logs/*'))
        games = sorted(list(map(GameParser, dirs)), key=lambda x: len(x.question_pairs), reverse=True)

        test = games[0::5]
        val = games[1::5]
        train = games[2::5]+games[3::5]+games[4::5]

        dataset_splits['test'] += [g.game_path for g in test]
        dataset_splits['validation'] += [g.game_path for g in val]
        dataset_splits['training'] += [g.game_path for g in train]
        json.dump(dataset_splits, open('config/dataset_splits_new.json','w'), indent=4)
        json.dump(dataset_splits, open('config/dataset_splits.json','w'), indent=4)


        dataset_splits['test'] = dataset_splits['test'][:2]
        dataset_splits['validation'] = dataset_splits['validation'][:2]
        dataset_splits['training'] = dataset_splits['training'][:2]
        json.dump(dataset_splits, open('config/dataset_splits_dev.json','w'), indent=4)
    
    dataset_splits = json.load(open(split_file))
    
    return dataset_splits

def onehot(x,n):
    retval = np.zeros(n)
    if x > 0:
        retval[x-1] = 1
    return retval

class GameParser:
    tokenizer = None
    model = None
    def __init__(self, game_path, load_dialogue=True, pov=0, intermediate=0, use_dialogue_moves=False):
        # print(game_path,end = ' ')
        self.load_dialogue = load_dialogue
        if pov not in (0,1,2,3,4):
            print('Point of view must be in (0,1,2,3,4), but got ', pov)
            exit()
        self.pov = pov
        self.use_dialogue_moves = use_dialogue_moves
        self.load_player1 = pov==1
        self.load_player2 = pov==2
        self.load_third_person = pov==3
        self.game_path = game_path
        # print(game_path)
        self.dialogue_file = glob(os.path.join(game_path,'mcc*log'))[0]
        self.questions_file = glob(os.path.join(game_path,'web*log'))[0]
        self.plan_file = glob(os.path.join(game_path,'plan*json'))[0]
        self.plan = json.load(open(self.plan_file))
        self.img_w = 96
        self.img_h = 96
        self.intermediate = intermediate
        
        self.flip_video = False
        for l in open(self.dialogue_file):
            if 'HAS JOINED' in l:
                player_name = l.strip().split()[1]
                self.flip_video = player_name[-1] == '2'
                break
        
        if  not os.path.isfile("config/materials.json") or \
            not os.path.isfile("config/mines.json") or \
            not os.path.isfile("config/tools.json"):
            plan_files = sorted(glob('data/*_logs/*/plan*.json'))
            materials = []
            tools = []
            mines = []
            for plan_file in plan_files:
                plan = json.load(open(plan_file))
                materials += plan['materials']
                tools += plan['tools']
                mines += plan['mines']
            materials = sorted(list(set(materials)))
            tools = sorted(list(set(tools)))
            mines = sorted(list(set(mines)))
            json.dump(materials, open('config/materials.json','w'), indent=4)
            json.dump(mines, open('config/mines.json','w'), indent=4)
            json.dump(tools, open('config/tools.json','w'), indent=4)
            
        materials = json.load(open('config/materials.json'))
        mines = json.load(open('config/mines.json'))
        tools = json.load(open('config/tools.json'))
        
        self.materials_dict = {x:i+1 for i,x in enumerate(materials)}
        self.mines_dict = {x:i+1 for i,x in enumerate(mines)}
        self.tools_dict = {x:i+1 for i,x in enumerate(tools)}
        
        self.__load_dialogue_act_labels()
        self.__load_dialogue_move_labels()
        self.__parse_dialogue()
        self.__parse_questions()
        self.__parse_start_end()
        self.__parse_question_pairs()
        self.__load_videos()
        self.__assign_dialogue_act_labels()
        self.__assign_dialogue_move_labels()
        self.__load_replay_data()
        self.__load_intermediate()
        
        # print(len(self.materials_dict))
        
        self.global_plan = []
        self.global_plan_mat = np.zeros((21,21))
        mine_counter = 0
        for n,v in zip(self.plan['materials'],self.plan['full']):
            if v['make']:
                mine = 0
                m1 = self.materials_dict[self.plan['materials'][v['make'][0][0]]]
                m2 = self.materials_dict[self.plan['materials'][v['make'][0][1]]]
                self.global_plan_mat[self.materials_dict[n]-1][m1-1] = 1
                self.global_plan_mat[self.materials_dict[n]-1][m2-1] = 1
            else:
                mine = self.mines_dict[self.plan['mines'][mine_counter]]
                mine_counter += 1
                m1 = 0
                m2 = 0
            mine = onehot(mine, len(self.mines_dict))
            m1 = onehot(m1,len(self.materials_dict))
            m2 = onehot(m2,len(self.materials_dict))
            mat = onehot(self.materials_dict[n],len(self.materials_dict))
            t = onehot(self.tools_dict[self.plan['tools'][v['tools'][0]]],len(self.tools_dict))
            step = np.concatenate((mat,m1,m2,mine,t))
            self.global_plan.append(step)
        
        self.player1_plan = []
        self.player1_plan_mat = np.zeros((21,21))
        mine_counter = 0
        for n,v in zip(self.plan['materials'],self.plan['player1']):
            if v['make']:
                mine = 0
                if v['make'][0][0] < 0:
                    m1 = 0
                    m2 = 0
                else:
                    m1 = self.materials_dict[self.plan['materials'][v['make'][0][0]]]
                    m2 = self.materials_dict[self.plan['materials'][v['make'][0][1]]]
                    self.player1_plan_mat[self.materials_dict[n]-1][m1-1] = 1
                    self.player1_plan_mat[self.materials_dict[n]-1][m2-1] = 1
            else:
                mine = self.mines_dict[self.plan['mines'][mine_counter]]
                mine_counter += 1
                m1 = 0
                m2 = 0
            mine = onehot(mine, len(self.mines_dict))
            m1 = onehot(m1,len(self.materials_dict))
            m2 = onehot(m2,len(self.materials_dict))
            mat = onehot(self.materials_dict[n],len(self.materials_dict))
            t = onehot(self.tools_dict[self.plan['tools'][v['tools'][0]]],len(self.tools_dict))
            step = np.concatenate((mat,m1,m2,mine,t))
            self.player1_plan.append(step)
        
        self.player2_plan = []
        self.player2_plan_mat = np.zeros((21,21))
        mine_counter = 0
        for n,v in zip(self.plan['materials'],self.plan['player2']):
            if v['make']:
                mine = 0
                if v['make'][0][0] < 0:
                    m1 = 0
                    m2 = 0
                else:
                    m1 = self.materials_dict[self.plan['materials'][v['make'][0][0]]]
                    m2 = self.materials_dict[self.plan['materials'][v['make'][0][1]]]
                    self.player2_plan_mat[self.materials_dict[n]-1][m1-1] = 1
                    self.player2_plan_mat[self.materials_dict[n]-1][m2-1] = 1
            else:
                mine = self.mines_dict[self.plan['mines'][mine_counter]]
                mine_counter += 1
                m1 = 0
                m2 = 0
            mine = onehot(mine, len(self.mines_dict))
            m1 = onehot(m1,len(self.materials_dict))
            m2 = onehot(m2,len(self.materials_dict))
            mat = onehot(self.materials_dict[n],len(self.materials_dict))
            t = onehot(self.tools_dict[self.plan['tools'][v['tools'][0]]],len(self.tools_dict))
            step = np.concatenate((mat,m1,m2,mine,t))
            self.player2_plan.append(step)
        # print(self.global_plan_mat.reshape(-1))
        # print(self.player1_plan_mat.reshape(-1))
        # print(self.player2_plan_mat.reshape(-1))
        # for x in zip(self.global_plan_mat.reshape(-1),self.player1_plan_mat.reshape(-1),self.player2_plan_mat.reshape(-1)):
        #     if sum(x) > 0:
        #         print(x)
        # exit()
        if self.load_player1:
            self.plan_repr = self.player1_plan_mat
            self.partner_plan = self.player2_plan_mat
        elif self.load_player2:
            self.plan_repr = self.player2_plan_mat
            self.partner_plan = self.player1_plan_mat
        else:
            self.plan_repr = self.global_plan_mat
            self.partner_plan = self.global_plan_mat
        self.global_diff_plan_mat = self.global_plan_mat - self.plan_repr
        self.partner_diff_plan_mat = self.global_plan_mat - self.partner_plan
        
        self.__iter_ts = self.start_ts
        
        self.action_labels = sorted([t for a in self.actions for t in a if t.PacketData in ['BlockChangeData']], key=lambda x: x.TickIndex)
        # for tick in ticks:
        #     print(int(tick.TickIndex/30), self.plan['materials'].index( tick.items[0]), int(tick.Name[-1]))
        # print(self.start_ts, self.end_ts, self.start_ts - self.end_ts, int(ticks[-1].TickIndex/30) if ticks else 0,self.action_file)
        # exit()
        self.materials = sorted(self.plan['materials'])
        
    def __len__(self):
        return self.end_ts - self.start_ts
            
    def __next__(self):
        if self.__iter_ts < self.end_ts:
            
            if self.load_dialogue:
                d = [x for x in self.dialogue_events if x[0] == self.__iter_ts]
                l = [x for x in self.dialogue_act_labels if x[0] == self.__iter_ts]
                d = d if d else None
                l = l if l else None
            else:
                d = None
                l = None

            if self.use_dialogue_moves:
                m = [x for x in self.dialogue_move_labels if x[0] == self.__iter_ts]
                m = m if m else None
            else:
                m = None

            if self.action_labels:
                a = [x for x in self.action_labels if (x.TickIndex//30 + self.start_ts) >= self.__iter_ts]
                if a:
                    try:
                        while not a[0].items:
                            a = a[1:]
                        al = self.materials.index(a[0].items[0]) if a else 0
                    except Exception:
                        print(a)
                        print(a[0])
                        print(a[0].items)
                        print(a[0].items[0])
                        exit()
                    at = a[0].TickIndex//30 + self.start_ts
                    an = int(a[0].Name[-1])
                    a = [(at,al,an)]
                else:
                    a = [(self.__iter_ts, self.materials.index(self.plan['materials'][0]), 1)]
                    a = None
            else:
                if self.end_ts - self.__iter_ts < 10:
                    # a = [(self.__iter_ts, self.materials.index(self.plan['materials'][0]), 1)]
                    a = None
                else:
                    a = None
            # if not self.__iter_ts % 30 == 0:
            #     a= None
            if not a is None:
                if not a[0][0] == self.__iter_ts:
                    a = None
            
            # q = [x for x in self.question_pairs if (x[0][0] < self.__iter_ts) and (x[0][1] > self.__iter_ts)]
            q = [x for x in self.question_pairs if (x[0][1] == self.__iter_ts)]
            q = q[0] if q else None
            frame_idx = self.__iter_ts - self.start_ts
            if self.load_third_person:
                frames = self.third_pers_frames
            elif self.load_player1:
                frames = self.player1_pov_frames
            elif self.load_player2:
                frames = self.player2_pov_frames
            else:
                frames = np.array([0])
            if len(frames) == 1:
                f = np.zeros((self.img_h,self.img_w,3))
            else:
                if frame_idx < frames.shape[0]:
                    f = frames[frame_idx]
                else:
                    f = np.zeros((self.img_h,self.img_w,3))
            if self.do_upperbound:
                if not q is None:
                    qnum = 0
                    base_rep = np.concatenate([
                        onehot(q[0][2],2),
                        onehot(q[0][3],2),
                        onehot(q[0][4][qnum][0]+1,2),
                        onehot(self.materials_dict[q[0][5][qnum][1]],len(self.materials_dict)),
                        onehot(q[0][4][qnum][0]+1,2),
                        onehot(self.materials_dict[q[0][5][qnum][1]],len(self.materials_dict)),
                        onehot(['YES','MAYBE','NO'].index(q[1][0][qnum])+1,3),
                        onehot(['YES','MAYBE','NO'].index(q[1][1][qnum])+1,3)
                    ])
                    base_rep = np.concatenate([base_rep, np.zeros(1024-base_rep.shape[0])])
                    ToM6 = base_rep if self.ToM6 is not None else np.zeros(1024)
                    qnum = 1
                    base_rep = np.concatenate([
                        onehot(q[0][2],2),
                        onehot(q[0][3],2),
                        onehot(q[0][4][qnum][0]+1,2),
                        onehot(self.materials_dict[q[0][5][qnum][1]],len(self.materials_dict)),
                        onehot(q[0][4][qnum][0]+1,2),
                        onehot(self.materials_dict[q[0][5][qnum][1]],len(self.materials_dict)),
                        onehot(['YES','MAYBE','NO'].index(q[1][0][qnum])+1,3),
                        onehot(['YES','MAYBE','NO'].index(q[1][1][qnum])+1,3)
                    ])
                    base_rep = np.concatenate([base_rep, np.zeros(1024-base_rep.shape[0])])
                    ToM7 = base_rep if self.ToM7 is not None else np.zeros(1024)
                    qnum = 2
                    base_rep = np.concatenate([
                        onehot(q[0][2],2),
                        onehot(q[0][3],2),
                        onehot(q[0][4][qnum]+1,2),
                        onehot(q[0][4][qnum]+1,2),
                        onehot(self.materials_dict[q[1][0][qnum]] if q[1][0][qnum] in self.materials_dict else len(self.materials_dict)+1,len(self.materials_dict)+1),
                        onehot(self.materials_dict[q[1][1][qnum]] if q[1][1][qnum] in self.materials_dict else len(self.materials_dict)+1,len(self.materials_dict)+1)
                    ])
                    base_rep = np.concatenate([base_rep, np.zeros(1024-base_rep.shape[0])])
                    ToM8 = base_rep if self.ToM8 is not None else np.zeros(1024)
                else:
                    ToM6 = np.zeros(1024)
                    ToM7 = np.zeros(1024)
                    ToM8 = np.zeros(1024)
                if not l is None:
                    base_rep = np.concatenate([
                        onehot(l[0][1],2),
                        onehot(l[0][2],len(self.dialogue_act_labels_dict))
                    ])
                    base_rep = np.concatenate([base_rep, np.zeros(1024-base_rep.shape[0])])
                    DAct = base_rep if self.DAct is not None else np.zeros(1024)
                else:
                    DAct = np.zeros(1024)
                if not m is None:
                    base_rep = np.concatenate([
                        onehot(m[0][1],2),
                        onehot(m[0][2][0],len(self.dialogue_move_labels_dict)),
                        onehot(m[0][2][1],len(self.tools_dict) + len(self.materials_dict) + len(self.mines_dict)+1),
                        onehot(m[0][2][2],len(self.tools_dict) + len(self.materials_dict) + len(self.mines_dict)+1),
                        onehot(m[0][2][3],len(self.tools_dict) + len(self.materials_dict) + len(self.mines_dict)+1),
                    ])
                    base_rep = np.concatenate([base_rep, np.zeros(1024-base_rep.shape[0])])
                    DMove = base_rep if self.DMove is not None else np.zeros(1024)
                else:
                    DMove = np.zeros(1024)
            else:
                ToM6 = self.ToM6[frame_idx] if self.ToM6 is not None else np.zeros(1024)
                ToM7 = self.ToM7[frame_idx] if self.ToM7 is not None else np.zeros(1024)
                ToM8 = self.ToM8[frame_idx] if self.ToM8 is not None else np.zeros(1024)
                DAct = self.DAct[frame_idx] if self.DAct is not None else np.zeros(1024)
                DMove = self.DAct[frame_idx] if self.DMove is not None else np.zeros(1024)
                # if not m is None:
                #     base_rep = np.concatenate([
                #         onehot(m[0][1],2),
                #         onehot(m[0][2][0],len(self.dialogue_move_labels_dict)),
                #         onehot(m[0][2][1],len(self.tools_dict) + len(self.materials_dict) + len(self.mines_dict)+1),
                #         onehot(m[0][2][2],len(self.tools_dict) + len(self.materials_dict) + len(self.mines_dict)+1),
                #         onehot(m[0][2][3],len(self.tools_dict) + len(self.materials_dict) + len(self.mines_dict)+1),
                #     ])
                #     base_rep = np.concatenate([base_rep, np.zeros(1024-base_rep.shape[0])])
                #     DMove = base_rep if self.DMove is not None else np.zeros(1024)
                # else:
                #     DMove = np.zeros(1024)
            intermediate = np.concatenate([ToM6,ToM7,ToM8,DAct,DMove])
            retval = ((self.__iter_ts,self.pov),d,l,q,f,a,intermediate,m)
            self.__iter_ts += 1
            return retval
        self.__iter_ts = self.start_ts
        raise StopIteration()
    
    def __iter__(self):
        return self
    
    def __load_videos(self):
        d = self.end_ts - self.start_ts
        
        if self.load_third_person:
            try:
                self.third_pers_file = glob(os.path.join(self.game_path,'third*gif'))[0]
                np_file = self.third_pers_file[:-3]+'npz'
                if os.path.isfile(np_file):
                    self.third_pers_frames = np.load(np_file)['data']
                else:
                    frames = imageio.get_reader(self.third_pers_file, '.gif')
                    reshaper  = lambda x: cv2.resize(x,(self.img_h,self.img_w))
                    if 'main' in self.game_path:
                        self.third_pers_frames = np.array([reshaper(f[95:4*95,250:-249,2::-1]) for f in frames])
                    else:
                        self.third_pers_frames = np.array([reshaper(f[-3*95:,250:-249,2::-1]) for f in frames])
                    print(np_file,end=' ')
                    np.savez_compressed(open(np_file,'wb'), data=self.third_pers_frames)
                    print('saved')
            except Exception as e:
                self.third_pers_frames = np.array([0])
                
            if self.third_pers_frames.shape[0]//d < 10:
                self.third_pov_frame_rate = 6
            else:
                if self.third_pers_frames.shape[0]//d < 20:
                    self.third_pov_frame_rate = 12
                else:
                    if self.third_pers_frames.shape[0]//d < 45:
                        self.third_pov_frame_rate = 30
                    else:
                        self.third_pov_frame_rate = 60
            self.third_pers_frames = self.third_pers_frames[::self.third_pov_frame_rate]
        else:
            self.third_pers_frames = np.array([0])
            
        if self.load_player1:
            try:
                search_str = 'play2*gif' if self.flip_video else 'play1*gif'
                self.player1_pov_file = glob(os.path.join(self.game_path,search_str))[0]
                np_file = self.player1_pov_file[:-3]+'npz'
                if os.path.isfile(np_file):
                    self.player1_pov_frames = np.load(np_file)['data']
                else:                
                    frames = imageio.get_reader(self.player1_pov_file, '.gif')
                    reshaper  = lambda x: cv2.resize(x,(self.img_h,self.img_w))
                    self.player1_pov_frames = np.array([reshaper(f[:,:,2::-1]) for f in frames])
                    print(np_file,end=' ')
                    np.savez_compressed(open(np_file,'wb'), data=self.player1_pov_frames)
                    print('saved')
            except Exception as e:
                self.player1_pov_frames = np.array([0])
            
            if self.player1_pov_frames.shape[0]//d < 10:
                self.player1_pov_frame_rate = 6
            else:
                if self.player1_pov_frames.shape[0]//d < 20:
                    self.player1_pov_frame_rate = 12
                else:
                    if self.player1_pov_frames.shape[0]//d < 45:
                        self.player1_pov_frame_rate = 30
                    else:
                        self.player1_pov_frame_rate = 60
            self.player1_pov_frames = self.player1_pov_frames[::self.player1_pov_frame_rate]
        else:
            self.player1_pov_frames = np.array([0])
            
        if self.load_player2:
            try:
                search_str = 'play1*gif' if self.flip_video else 'play2*gif'
                self.player2_pov_file = glob(os.path.join(self.game_path,search_str))[0]
                np_file = self.player2_pov_file[:-3]+'npz'
                if os.path.isfile(np_file):
                    self.player2_pov_frames = np.load(np_file)['data']
                else:
                    frames = imageio.get_reader(self.player2_pov_file, '.gif')
                    reshaper  = lambda x: cv2.resize(x,(self.img_h,self.img_w))
                    self.player2_pov_frames = np.array([reshaper(f[:,:,2::-1]) for f in frames])
                    print(np_file,end=' ')
                    np.savez_compressed(open(np_file,'wb'), data=self.player2_pov_frames)
                    print('saved')
            except Exception as e:
                self.player2_pov_frames = np.array([0])
                
            if self.player2_pov_frames.shape[0]//d < 10:
                self.player2_pov_frame_rate = 6
            else:
                if self.player2_pov_frames.shape[0]//d < 20:
                    self.player2_pov_frame_rate = 12
                else:
                    if self.player2_pov_frames.shape[0]//d < 45:
                        self.player2_pov_frame_rate = 30
                    else:
                        self.player2_pov_frame_rate = 60
            self.player2_pov_frames = self.player2_pov_frames[::self.player2_pov_frame_rate]
        else:
            self.player2_pov_frames = np.array([0])

    def __parse_question_pairs(self):
        question_dict = {}        
        for q in self.questions:
            k = q[2][0][1] + q[2][1][1]
            if not k in question_dict:
                question_dict[k] = []
            question_dict[k].append(q)
        
        self.question_pairs = []
        for k,v in question_dict.items():
            if len(v) == 2:
                if v[0][1]+v[1][1] == 3:
                    self.question_pairs.append(v)
            else:
                while len(v) > 1:
                    pair = []
                    pair.append(v.pop(0))
                    pair.append(v.pop(0))
                    while not pair[0][1]+pair[1][1] == 3:
                        if not v:
                            break
                        # print(game_path,pair)
                        pair.append(v.pop(0))
                        pair.pop(0)
                        if not v:
                            break
                    self.question_pairs.append(pair)
        self.question_pairs = sorted(self.question_pairs, key=lambda x: x[0][0])
        if self.load_player2 or self.pov==4:
            self.question_pairs = [sorted(q, key=lambda x: x[1],reverse=True) for q in self.question_pairs]
        else:
            self.question_pairs = [sorted(q, key=lambda x: x[1]) for q in self.question_pairs]
        
        
        self.question_pairs = [((a[0], b[0], a[1], b[1], a[2], b[2]), (a[3], b[3])) for a,b in self.question_pairs]

    def __parse_dialogue(self):
        self.dialogue_events = []
        # if not self.load_dialogue:
        #     return 
        save_path = os.path.join(self.game_path,f'dialogue_{self.game_path.split("/")[-1]}.pkl')
        # print(save_path)
        # exit()
        if os.path.isfile(save_path):
            self.dialogue_events = pickle.load(open( save_path, "rb" ))
            return
        for x in open(self.dialogue_file):
            if '[Async Chat Thread' in x:
                ts = list(map(int,x.split(' [')[0].strip('[]').split(':')))
                ts = 3600*ts[0] + 60*ts[1] + ts[2]
                player, event = x.strip().split('/INFO]: []<sledmcc')[1].split('> ',1)
                event = event.lower()
                event = ''.join([x if x in string.ascii_lowercase else f' {x} ' for x in event]).strip()
                event = event.replace('  ',' ').replace('  ',' ')
                player = int(player)
                if GameParser.tokenizer is None:
                    GameParser.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
                if self.model is None:
                    GameParser.model = BertModel.from_pretrained('bert-large-uncased', output_hidden_states=True).to(DEVICE)
                encoded_dict = GameParser.tokenizer.encode_plus(
                    event,  # Sentence to encode.
                    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                    return_tensors='pt',  # Return pytorch tensors.
                )
                token_ids = encoded_dict['input_ids'].to(DEVICE)
                segment_ids = torch.ones(token_ids.size()).long().to(DEVICE)
                GameParser.model.eval()
                with torch.no_grad():
                    outputs = GameParser.model(input_ids=token_ids, token_type_ids=segment_ids)
                outputs = outputs[1][0].cpu().data.numpy()
                self.dialogue_events.append((ts,player,event,outputs))
        pickle.dump(self.dialogue_events, open( save_path, "wb" ))
        print(f'Saved to {save_path}',flush=True)
        
    def __parse_questions(self):
        self.questions = []
        for x in open(self.questions_file):
            if x[0] == '#':
                ts, qs = x.strip().split(' Number of records inserted: 1 # player')
                # print(ts,qs)
                
                ts = list(map(int,ts.split(' ')[5].split(':')))
                ts = 3600*ts[0] + 60*ts[1] + ts[2]
                
                player = int(qs[0])
                questions = qs[2:].split(';')
                answers =[x[7:] for x in questions[3:]]
                questions = [x[9:].split(' ') for x in questions[:3]]
                questions[0] = (int(questions[0][0] == 'Have'), questions[0][-3])
                questions[1] = (int(questions[1][2] == 'know'), questions[1][-1])
                questions[2] = int(questions[2][1] == 'are')
                
                self.questions.append((ts,player,questions,answers))
    def __parse_start_end(self):        
        self.start_ts = [x.strip() for x in open(self.dialogue_file) if 'THEY ARE PLAYER' in x][1]
        self.start_ts = list(map(int,self.start_ts.split('] [')[0][1:].split(':')))
        self.start_ts = 3600*self.start_ts[0] + 60*self.start_ts[1] + self.start_ts[2]
        try:
            self.start_ts = max(self.start_ts, self.questions[0][0]-75)
        except Exception as e:
            pass
        
        self.end_ts = [x.strip() for x in open(self.dialogue_file) if 'Stopping' in x]
        if self.end_ts:
            self.end_ts = self.end_ts[0]
            self.end_ts = list(map(int,self.end_ts.split('] [')[0][1:].split(':')))
            self.end_ts = 3600*self.end_ts[0] + 60*self.end_ts[1] + self.end_ts[2]
        else:
            self.end_ts = self.dialogue_events[-1][0]
        try:
            self.end_ts = max(self.end_ts, self.questions[-1][0]) + 1
        except Exception as e:
            pass
        
    def __load_dialogue_act_labels(self):
        file_name = 'config/dialogue_act_labels.json'
        if not os.path.isfile(file_name):
            files = sorted(glob('/home/*/MCC/*done.txt'))
            dialogue_act_dict = {}
            for file in files:
                game_str = ''
                for line in open(file):
                    line  = line.strip()
                    if '_logs/' in line:
                        game_str = line
                    else:
                        if line:
                            line = line.split()
                            key = f'{game_str}#{line[0]}'
                            dialogue_act_dict[key] = line[-1]
            json.dump(dialogue_act_dict,open(file_name,'w'), indent=4)
        self.dialogue_act_dict = json.load(open(file_name))
        self.dialogue_act_labels_dict = {l : i for i, l in enumerate(sorted(list(set(self.dialogue_act_dict.values()))))}
        self.dialogue_act_bias = {l : sum([int(x==l) for x in self.dialogue_act_dict.values()]) for l in self.dialogue_act_labels_dict.keys()}
        json.dump(self.dialogue_act_labels_dict,open('config/dialogue_act_label_names.json','w'), indent=4)
        # print(self.dialogue_act_bias)
        # print(self.dialogue_act_labels_dict)
        # exit()
        
    def __assign_dialogue_act_labels(self):
        
        log_file = glob('/'.join([self.game_path,'mcc*log']))[0][5:]
        self.dialogue_act_labels = []
        for emb in self.dialogue_events:
            ts = emb[0]
            h = ts//3600
            m = (ts%3600)//60
            s = ts%60
            key = f'{log_file}#[{h:02d}:{m:02d}:{s:02d}]:{emb[1]}>'
            self.dialogue_act_labels.append((emb[0],emb[1],self.dialogue_act_labels_dict[self.dialogue_act_dict[key]]))
        
    def __load_dialogue_move_labels(self):
        file_name = "config/dialogue_move_labels.json"
        dialogue_move_dict = {}
        if not os.path.isfile(file_name):
            file_text = ''
            dialogue_moves = set()
            for line in open("/home/cpbara/MCC/dialogue_move_labels_final.txt"):
                line = line.strip()
                if not line:
                    continue
                if line[0] == '#':
                    continue
                if line[0] == '[':
                    tag_text = glob(f'data/*/*/mcc_{file_text}.log')[0].split('/',1)[-1]
                    key = f'{tag_text}#{line.split()[0]}'
                    value = line.split()[-1].split('#') 
                    if len(value) < 4:
                        value += ['IGNORE']*(4-len(value))
                    dialogue_moves.add(value[0])
                    value = '#'.join(value)
                    dialogue_move_dict[key] = value
                    # print(key,value)
                    # break
                else:
                    file_text = line
                # print(line)
            dialogue_moves = sorted(list(dialogue_moves))
            # print(dialogue_moves)
                
            json.dump(dialogue_move_dict,open(file_name,'w'), indent=4)
        self.dialogue_move_dict = json.load(open(file_name))
        self.dialogue_move_labels_dict = {l : i for i, l in enumerate(sorted(list(set([lbl.split('#')[0] for lbl in self.dialogue_move_dict.values()]))))}
        self.dialogue_move_bias = {l : sum([int(x==l) for x in self.dialogue_move_dict.values()]) for l in self.dialogue_move_labels_dict.keys()}
        json.dump(self.dialogue_move_labels_dict,open('config/dialogue_move_label_names.json','w'), indent=4)
        
    def __assign_dialogue_move_labels(self):
        
        log_file = glob('/'.join([self.game_path,'mcc*log']))[0][5:]
        self.dialogue_move_labels = []
        for emb in self.dialogue_events:
            ts = emb[0]
            h = ts//3600
            m = (ts%3600)//60
            s = ts%60
            key = f'{log_file}#[{h:02d}:{m:02d}:{s:02d}]:{emb[1]}>'
            move = self.dialogue_move_dict[key].split('#')
            move[0] = self.dialogue_move_labels_dict[move[0]]
            for i,m in enumerate(move[1:]):
                if m == 'IGNORE':
                    move[i+1] = 0
                elif m in self.materials_dict:
                    move[i+1] = self.materials_dict[m]
                elif m in self.mines_dict:
                    move[i+1] = self.mines_dict[m] + len(self.materials_dict)
                elif m in self.tools_dict:
                    move[i+1] = self.tools_dict[m] + len(self.materials_dict) + len(self.mines_dict)
                else:
                    print(move)
                    exit()
            # print(move,self.dialogue_move_dict[key],key)
            # exit()
            self.dialogue_move_labels.append((emb[0],emb[1],move))
            
    def __load_replay_data(self):
        self.action_file = "data/ReplayData/ActionsData_mcc_" + self.game_path.split('/')[-1]
        with open(self.action_file) as f:
            data  = ' '.join(x.strip() for x in f).split('action')
            # preface = data[0]
            self.actions = list(map(proc_action, data[1:]))
            
    def __load_intermediate(self):
        if self.intermediate > 15:
            self.do_upperbound = True
        else:
            self.do_upperbound = False
        if self.pov in [1,2]:
            self.ToM6 = np.load(glob(f'{self.game_path}/intermediate_ToM6*player{self.pov}.npz')[0])['data'] if self.intermediate % 2 else None
            self.intermediate = self.intermediate // 2
            self.ToM7 = np.load(glob(f'{self.game_path}/intermediate_ToM7*player{self.pov}.npz')[0])['data'] if self.intermediate % 2 else None
            self.intermediate = self.intermediate // 2
            self.ToM8 = np.load(glob(f'{self.game_path}/intermediate_ToM8*player{self.pov}.npz')[0])['data'] if self.intermediate % 2 else None
            self.intermediate = self.intermediate // 2
            self.DAct = np.load(glob(f'{self.game_path}/intermediate_DAct*player{self.pov}.npz')[0])['data'] if self.intermediate % 2 else None
            self.intermediate = self.intermediate // 2
            self.DMove = None
            # print(self.ToM6)
            # print(self.ToM7)
            # print(self.ToM8)
            # print(self.DAct)
        else:
            self.ToM6 = None
            self.ToM7 = None
            self.ToM8 = None
            self.DAct = None
            self.DMove = None
        # exit()

