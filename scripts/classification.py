import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from sklearn import model_selection, naive_bayes, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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

    
    (train_x, train_y) = getPointsForGames(train_games, data)
    (test_x, test_y) = getPointsForGames(test_games, data)


    #test_x = np.array([np.array(xi) for xi in pointsAll])
    Naive = naive_bayes.MultinomialNB()
    Naive.fit(train_x, train_y)

    predictions_NB = Naive.predict(test_x)
    print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, test_y)*100)

    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(train_x, train_y)

    predictions_SVM = SVM.predict(test_x)    
    print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, test_y)*100)



    print('Kraj')

def getPointsForGames(games, data):

    A = np.zeros(12) #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    outcomesAll = []
    for game in games:
        is_game = data['GAME_ID'] == game
        subset = data[is_game]
        subset = subset.sort_values(by=['LOCATION'], ascending=False)
        home = subset[subset.LOCATION == 'H']
        #away = subset[subset.LOCATION == 'A']
        if len(home) < 12:
            for i in range(12 - len(home)):
                new_row = pd.DataFrame({'GAME_ID' : [home['GAME_ID'].iloc[0]], 
                                        'LOCATION' : ['H'], 
                                        'W/L' : [home['W/L'].iloc[0]], 
                                        'DEFENSIVE_RATING' : [home['DEFENSIVE_RATING'].iloc[0]], 
                                        'PLAYER_NAME' : [''], 
                                        'PLAYER_ID' : [i], 
                                        'POINTS' : [0], 
                                        'TEAM' : [home['TEAM'].iloc[0]], 
                                        'OPPOSING_TEAM' : [home['OPPOSING_TEAM'].iloc[0]], 
                                        'OUTCOME' : [home['OUTCOME'].iloc[0]]})
                home = pd.concat([home, new_row]).reset_index(drop=True)

        '''
        if len(away) < 12:
            for i in range(12 - len(away)):
                new_row = pd.DataFrame({'GAME_ID' : [away['GAME_ID'].iloc[0]], 
                                        'LOCATION' : ['A'], 
                                        'W/L' : [away['W/L'].iloc[0]], 
                                        'DEFENSIVE_RATING' : [away['DEFENSIVE_RATING'].iloc[0]], 
                                        'PLAYER_NAME' : [''], 
                                        'PLAYER_ID' : [i], 
                                        'POINTS' : [0], 
                                        'TEAM' : [away['TEAM'].iloc[0]], 
                                        'OPPOSING_TEAM' : [away['OPPOSING_TEAM'].iloc[0]], 
                                        'OUTCOME' : [away['OUTCOME'].iloc[0]]})
                away = pd.concat([away, new_row]).reset_index(drop=True)
        '''
        points = home.POINTS.tolist() #+ away.POINTS.tolist()
        new_row = np.array(points)
        outcome = home.OUTCOME.iloc[0]
        #pointsAll.append(points)
        outcomesAll.append(outcome)
        A = np.vstack([A, new_row])
        #A = np.append(A, new_row)
        '''
        print(home)
        print(len(away))
        print(len(subset))
        '''
    A = np.delete(A, (0), axis=0)
    return (A, outcomesAll)

if __name__ == "__main__":
    runScript()
