import json
import pickle

with open('teams_players.json') as json_file:  
    players = json.load(json_file)
    print(players['TOR'])

rModels = pickle.load(open('elastic_models.pkl', 'rb'))
