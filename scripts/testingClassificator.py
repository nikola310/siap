import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from process_players import fix_team_name

with open('teams_players.json') as json_file:  
    players = json.load(json_file)

rModels = pickle.load(open('elastic_models.pkl', 'rb'))
classifier = pickle.load(open('nb_classifier.pkl', 'rb'))

data = pd.read_csv("../data/season1516processed.csv")

real_outcome = np.array([0])
predicted_outcome = np.array([0])

for row in data.iterrows():
    home_team = fix_team_name(row[1]['HOME'])
    defensive_rating = row[1]['V_DEFRTG']
    away_team = row[1]['VISITOR']
    home_points = row[1]['H_PTS']
    away_points = row[1]['V_PTS']

    outcome = 1 if (home_points - away_points) > 0 else 0

    predictions = []
    j = 0
    for i, player in enumerate(players[home_team]):
        if j == 11:
            break
        score = np.round(rModels[player].predict(np.array([defensive_rating]).reshape(-1, 1))[0])
        predictions.append(score)
        j += 1

    if len(predictions) < 12:
        for i in range(12 - len(predictions)):
            predictions.append(0)

    #print('Prediction ' + str(classifier.predict(np.array(predictions).reshape(1, -1))))
    #print('Real ' + str(outcome))
    pred = classifier.predict(np.array(predictions).reshape(1, -1))
    real_outcome = np.vstack([real_outcome, pred])
    predicted_outcome = np.vstack([predicted_outcome, outcome])

real_outcome = np.delete(real_outcome, (0), axis=0)
predicted_outcome = np.delete(predicted_outcome, (0), axis=0)

print("Naive Bayes Accuracy Score -> ", accuracy_score(predicted_outcome, real_outcome)*100)
