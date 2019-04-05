import pickle

import numpy as np
import pandas as pd
from sklearn import naive_bayes, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def runScript():
    #np.random.seed(500)
    data = pd.read_csv("../data/dataSet_processed.csv")
    games = data.GAME_ID.unique()

    train_games, test_games = train_test_split(games, test_size=0.3)

    encoder = LabelEncoder()
    data['OUTCOME'] = encoder.fit_transform(data['W/L'])


    (train_x, train_y) = get_points_for_games(train_games, data)
    (test_x, test_y) = get_points_for_games(test_games, data)

    #test_x = np.array([np.array(xi) for xi in pointsAll])
    naive = naive_bayes.MultinomialNB()
    naive.fit(train_x, train_y)

    predictions_nb = naive.predict(test_x)
    print("Naive Bayes Accuracy Score -> ", f1_score(predictions_nb, test_y)*100)

    svm_model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    svm_model.fit(train_x, train_y)

    predictions_svm = svm_model.predict(test_x)
    print("SVM Accuracy Score -> ", f1_score(predictions_svm, test_y)*100)

    pickle.dump(naive, open('nb_classifier.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(naive, open('svm_classifier.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def get_points_for_games(games, data):

    x_data = np.zeros(24) #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    y_data = []
    for game in games:
        is_game = data['GAME_ID'] == game
        subset = data[is_game]
        subset = subset.sort_values(by=['LOCATION'], ascending=False)
        home = subset[subset.LOCATION == 'H']
        away = subset[subset.LOCATION == 'A']
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

        points = home.POINTS.tolist() + away.POINTS.tolist()
        new_row = np.array(points)
        outcome = home.OUTCOME.iloc[0]
        #pointsAll.append(points)
        y_data.append(outcome)
        x_data = np.vstack([x_data, new_row])

    x_data = np.delete(x_data, (0), axis=0)
    return (x_data, y_data)

if __name__ == "__main__":
    runScript()
