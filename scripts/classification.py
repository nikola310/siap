import pickle

import numpy as np
import pandas as pd
from sklearn import naive_bayes, svm
from sklearn.metrics import f1_score, log_loss
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PLAYER_NUM = 10

def run_script():
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
    print("Naive Bayes Accuracy Score -> ", f1_score(test_y, predictions_nb)*100)

    svm_model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    svm_model.fit(train_x, train_y)

    predictions_svm = svm_model.predict(test_x)
    print("SVM Accuracy Score -> ", f1_score(test_y, predictions_svm)*100)

    pickle.dump(naive, open('nb_classifier.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(naive, open('svm_classifier.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    print('Stacking and majority voting start')
    naive_s = naive_bayes.MultinomialNB()
    svm_model_s = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    svm_model_s.probability = True
    soft = VotingClassifier(estimators=[('naive', naive_s), ('svm', svm_model_s)], 
                            weights=[1, 3], voting='soft')
    soft.fit(train_x, train_y)
    predictions_soft = soft.predict(test_x)
    print('Log loss: ', log_loss(test_y, predictions_soft))
    print("Voting Accuracy Score -> ", f1_score(test_y, predictions_soft)*100)
    naive_h = naive_bayes.MultinomialNB()
    svm_model_h = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    
    hard = VotingClassifier(estimators=[('naive', naive_h), ('svm', svm_model_h)], 
                            weights=[1, 3], voting='hard')
    hard.fit(train_x, train_y)
    predictions_hard = hard.predict(test_x)
    print('Log loss: ', log_loss(test_y, predictions_hard))
    print("Voting Accuracy Score -> ", f1_score(test_y, predictions_hard)*100)

    pickle.dump(soft, open('soft.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(hard, open('hard.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def get_points_for_games(games, data):

    x_data = np.zeros(2*PLAYER_NUM)
    y_data = []
    for game in games:
        is_game = data['GAME_ID'] == game
        subset = data[is_game]
        subset = subset.sort_values(by=['LOCATION'], ascending=False)
        home = subset[subset.LOCATION == 'H']
        away = subset[subset.LOCATION == 'A']
        if len(home) < PLAYER_NUM:
            for i in range(PLAYER_NUM - len(home)):
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

        if len(away) < PLAYER_NUM:
            for i in range(PLAYER_NUM - len(away)):
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

        points = remove_extra_elements(home.POINTS.tolist()) + remove_extra_elements(away.POINTS.tolist())
        new_row = np.array(points)
        outcome = home.OUTCOME.iloc[0]
        #pointsAll.append(points)
        y_data.append(outcome)
        x_data = np.vstack([x_data, new_row])

    x_data = np.delete(x_data, (0), axis=0)
    return (x_data, y_data)

def remove_extra_elements(l):
    if len(l) - PLAYER_NUM > 0:
        n = len(l) - PLAYER_NUM
        return l[:-n or None]
    else:
        return l

if __name__ == "__main__":
    run_script()
