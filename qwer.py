import json, os
from glob import glob
from src.data.game_parser import GameParser

dialogue_act_labels_file = "config/dialogue_act_labels.json"
dialogue_act_labels = json.load(open(dialogue_act_labels_file))

# for x in dialogue_act_labels.items():
#     print(x)
#     break

act_label_files = glob("/home/cpbara/MCC/dialogue_labeling/*done.txt")

file_text = ''
for act_label_file in act_label_files:
    for line in open(act_label_file):
        line = line.strip()
        if line:
            if line[0] == '[':
                key = f'{file_text}#{line.split()[0]}'
                value = line.split()[-1]
                if key not in dialogue_act_labels: 
                    dialogue_act_labels[key] = value
                
            else:
                file_text = line
json.dump(dialogue_act_labels, open(dialogue_act_labels_file,'w'))

dialogue_move_labels_file = "config/dialogue_move_labels.json"
dialogue_move_labels = {}

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
        dialogue_move_labels[key] = value
        # print(key,value)
        # break
    else:
        file_text = line
    # print(line)
dialogue_moves = sorted(list(dialogue_moves))
# print(dialogue_moves)
    
json.dump(dialogue_move_labels, open(dialogue_move_labels_file,'w'))
json.dump({k:v+1 for v,k in enumerate(dialogue_moves)}, open('config/dialogue_move_label_names.json','w'))

# exit()

if not os.path.isfile('config/dataset_splits.json'):
    # print('data/*_logs/*')
    dirs = sorted(glob('data/*_logs/*'))
    # for d in dirs:
    #     print(d)
    # exit()
    games = sorted(list(map(GameParser, dirs)), key=lambda x: len(x.question_pairs), reverse=True)

    test = games[0::5]
    val = games[1::5]
    train = games[2::5]+games[3::5]+games[4::5]

    dataset_splits = {'test' : [g.game_path for g in test], 'validation' : [g.game_path for g in val], 'training' : [g.game_path for g in train]}
    json.dump(dataset_splits, open('config/dataset_splits.json','w'))
else:
    dataset_splits = json.load(open('config/dataset_splits.json'))
    
for key, game_lst in dataset_splits.items():
    for game_dir in game_lst:
        print(len(game_lst), key, game_dir)
        GameParser(game_dir, pov=0)
        GameParser(game_dir, pov=1)
        GameParser(game_dir, pov=2)