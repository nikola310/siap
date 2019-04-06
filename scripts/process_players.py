import json

import pandas as pd

def run_script():
    data = pd.read_csv("../data/players.csv")

    og_data = pd.read_csv("../data/dataSet_processed.csv")

    player_names = og_data.PLAYER_NAME.unique()

    nseason = data.Player.unique()

    for index, name in enumerate(nseason):
        nseason[index] = name.split('\\')[0].lower()

    player_diff = set()
    for name in player_names:
        if name not in nseason:
            player_diff.add(name)

    teams_players = {}
    for row in data.iterrows():
        player_name = row[1]['Player'].split('\\')[0].lower()
        team = row[1]['Tm']
        if player_name in player_names:
            if team in teams_players:
                teams_players[team].add(player_name)
            else:
                teams_players[team] = set()
                teams_players[team].add(player_name)

    for row in og_data.iterrows():
        player_name = row[1]['PLAYER_NAME']
        team = fix_team_name(row[1]['TEAM'])
        if player_name in player_diff:
            #if team in teams_players:
            teams_players[team].add(player_name)
            #else:
            #    teams_players[team] = set()
            #    teams_players[team].add(player_name)

    with open('teams_players.json', 'w') as file:
        json.dump(teams_players, file, sort_keys=True, indent=4, default=set_default)

def fix_team_name(team):
    if team == 'CHA':
        return 'CHO'
    elif team == 'PHX':
        return 'PHO'
    elif team == 'BKN':
        return 'BRK'
    else:
        return team


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

if __name__ == "__main__":
    run_script()