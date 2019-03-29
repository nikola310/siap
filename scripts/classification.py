import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import json
import pylab as pl



def runScript():
    data = pd.read_csv("../data/dataSet_processed.csv")
    #train, test = train_test_split(data, test_size=0.3)
    players = data.PLAYER_ID.unique()
    games = data.GAME_ID.unique()

    train = {}
    test = {}

    train_games, test_games = train_test_split(games, test_size=0.3)

    Encoder = LabelEncoder()
    data['OUTCOME'] = Encoder.fit_transform(data['W/L'])

    out_df = pd.DataFrame()
    pointsAll = []
    outcomesAll = []
    for game in games:
        is_game = data['GAME_ID'] == game
        subset = data[is_game]
        subset = subset.sort_values(by=['LOCATION'], ascending=False)
        home_playas = subset[subset.LOCATION == 'H']
        away_playas = subset[subset.LOCATION == 'A']
        if len(home_playas) < 10:
            for i in range(10 - len(home_playas)):
                new_row = pd.DataFrame({'GAME_ID' : [home_playas['GAME_ID'].iloc[0]], 
                                        'LOCATION' : ['H'], 
                                        'W/L' : [home_playas['W/L'].iloc[0]], 
                                        'DEFENSIVE_RATING' : [home_playas['DEFENSIVE_RATING'].iloc[0]], 
                                        'PLAYER_NAME' : [''], 
                                        'PLAYER_ID' : [i], 
                                        'POINTS' : [0], 
                                        'TEAM' : [home_playas['TEAM'].iloc[0]], 
                                        'OPPOSING_TEAM' : [home_playas['OPPOSING_TEAM'].iloc[0]], 
                                        'OUTCOME' : [home_playas['OUTCOME'].iloc[0]]})
                home_playas = pd.concat([home_playas, new_row]).reset_index(drop=True)

        if len(away_playas) < 10:
            for i in range(10 - len(away_playas)):
                new_row = pd.DataFrame({'GAME_ID' : [away_playas['GAME_ID'].iloc[0]], 
                                        'LOCATION' : ['A'], 
                                        'W/L' : [away_playas['W/L'].iloc[0]], 
                                        'DEFENSIVE_RATING' : [away_playas['DEFENSIVE_RATING'].iloc[0]], 
                                        'PLAYER_NAME' : [''], 
                                        'PLAYER_ID' : [i], 
                                        'POINTS' : [0], 
                                        'TEAM' : [away_playas['TEAM'].iloc[0]], 
                                        'OPPOSING_TEAM' : [away_playas['OPPOSING_TEAM'].iloc[0]], 
                                        'OUTCOME' : [away_playas['OUTCOME'].iloc[0]]})
                away_playas = pd.concat([away_playas, new_row]).reset_index(drop=True)

        points = home_playas.POINTS.tolist() + away_playas.POINTS.tolist()
        outcome = home_playas.OUTCOME.iloc[0]
        pointsAll.append(points)
        outcomesAll.append(outcome)

        '''
        print(home_playas)
        print(len(away_playas))
        print(len(subset))
        '''

    test_x = np.array([np.array(xi) for xi in pointsAll])
    Naive = naive_bayes.MultinomialNB()
    Naive.fit(test_x, outcomesAll)

    print('Kraj')

if __name__ == "__main__":
    runScript()