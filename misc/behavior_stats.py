from glob import glob
import os
from src.data.game_parser import GameParser
import numpy as np

game_paths = glob('data/*_logs/*')

scores_dict = {}
for game_path in game_paths:
    game_scores = [[], [], [], []]
    game = GameParser(game_path)
    for q in game.question_pairs:
        # print(q[-1])
        game_scores[0].append(int(q[-1][0][0] == q[-1][1][0]))
        game_scores[1].append(int(q[-1][0][1] == q[-1][1][1]))
        game_scores[2].append(int(q[-1][0][2] == q[-1][1][2]))
        game_scores[3]. append(np.mean([game_scores[0][-1],game_scores[1][-1],game_scores[2][-1]]))
    if not game_scores[0]:
        continue
    scores_dict[game_path] = (list(map(np.mean,game_scores)))
num_games = len(scores_dict)

for task in range(4):
    print('Task:', task)
    sorted_games = [(key, val) for key, val in sorted(scores_dict.items(), key=lambda x: x[1][task])]
    bottom_quantile = sorted_games[:num_games//4+1]
    top_quantile = sorted_games[-num_games//4:]
    for prompt, quantile in [('bottom quantile',bottom_quantile),('top quantile',top_quantile)]:
        print(prompt)
        lengths = []
        moves = {}
        game_moves = {}
        num_nodes = []
        utterances = []
        for game_path,_ in quantile:
            game = GameParser(game_path)
            lengths.append(game.end_ts-game.start_ts)
            num_nodes.append(len(game.plan['full']))
            utterances.append(len(game.dialogue_events))
            for move in game.dialogue_move_labels:
                move_id = move[-1][0]
                if not move_id in moves:
                    moves[move_id] = 0
                moves[move_id] += 1
                if not game_path in game_moves:
                    game_moves[game_path] = []
                game_moves[game_path].append(move_id)
        print(f'{np.mean(utterances):04f}, {np.std(utterances):04f}',end=', ')
        print(f'{np.mean(lengths):04f}, {np.std(lengths):04f}',end=', ')
        lengths = sorted(lengths)
        print(f'[{lengths[0]} | {np.mean(lengths[9:10]):04f} | {np.mean(lengths[19:20]):04f} | {np.mean(lengths[29:30]):04f} | {lengths[-1]}]',end=', ')
        print(f'{np.mean(num_nodes):04f}, {np.std(num_nodes):04f}')
        keys = list(range(35))
        for key in keys:
            print(key,end=',\t')
        print()
        for key in keys:
            if key in moves:
                print(f'{moves[key]/sum(moves.values()):0.4f}',end=',\t')
            else:
                print(0,end=',\t')
        print()

        aux_fun = lambda key, lst: [sum([int(v == key) for v in val])/len(val) for _, val in lst.items()]
        box_plot_fun = lambda k, x: [
            np.quantile(aux_fun(k,x),0.1), 
            np.quantile(aux_fun(k,x),0.25), 
            np.quantile(aux_fun(k,x),0.5), 
            np.quantile(aux_fun(k,x),0.75), 
            np.quantile(aux_fun(k,x),0.9)
            ]

        print( 0, ',', ', '.join(map(lambda s: f"{s:0.3f}", box_plot_fun( 0, game_moves))))
        print( 6, ',', ', '.join(map(lambda s: f"{s:0.3f}", box_plot_fun( 6, game_moves))))
        print( 8, ',', ', '.join(map(lambda s: f"{s:0.3f}", box_plot_fun( 8, game_moves))))
        print(19, ',', ', '.join(map(lambda s: f"{s:0.3f}", box_plot_fun(19, game_moves))))
        print(31, ',', ', '.join(map(lambda s: f"{s:0.3f}", box_plot_fun(31, game_moves))))

        print( 0, ',', ', '.join(map(lambda s: f"{s:0.3f}", aux_fun( 0, game_moves))))
        print( 6, ',', ', '.join(map(lambda s: f"{s:0.3f}", aux_fun( 6, game_moves))))
        print( 8, ',', ', '.join(map(lambda s: f"{s:0.3f}", aux_fun( 8, game_moves))))
        print(19, ',', ', '.join(map(lambda s: f"{s:0.3f}", aux_fun(19, game_moves))))
        print(31, ',', ', '.join(map(lambda s: f"{s:0.3f}", aux_fun(31, game_moves))))


