import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from process_players import fix_team_name


def get_outcome(data):
    real_outcome = np.array([0])

    for row in data.iterrows():
        home_points = row[1]['H_PTS']
        away_points = row[1]['V_PTS']

        outcome = 1 if (home_points - away_points) > 0 else 0

        real_outcome = np.vstack([real_outcome, outcome])

    real_outcome = np.delete(real_outcome, (0), axis=0)
    return real_outcome

def get_predictions(data, classifier, models, players):
    predicted_outcome = np.array([0])

    for row in data.iterrows():
        home_team = fix_team_name(row[1]['HOME'])
        visiting_team = fix_team_name(row[1]['VISITOR'])
        defensive_rating = row[1]['V_DEFRTG']
        defensive_rating_home = row[1]['H_DEFRTG']

        predictions_home = []
        j = 0
        for player in players[home_team]:
            if j == 11:
                break
            score = np.round(models[player].predict(np.array([defensive_rating]).reshape(-1, 1))[0])
            predictions_home.append(score)
            j += 1

        if len(predictions_home) < 12:
            for _ in range(12 - len(predictions_home)):
                predictions_home.append(0)

        predictions_away = []
        j = 0
        for player in players[home_team]:
            if j == 11:
                break
            score = np.round(models[player].predict(
                np.array([defensive_rating_home]).reshape(-1, 1))[0])
            predictions_away.append(score)
            j += 1

        if len(predictions_away) < 12:
            for _ in range(12 - len(predictions_away)):
                predictions_away.append(0)

        predictions = predictions_home + predictions_away

        pred = classifier.predict(np.array(predictions).reshape(1, -1))
        predicted_outcome = np.vstack([predicted_outcome, pred])

    predicted_outcome = np.delete(predicted_outcome, (0), axis=0)
    return predicted_outcome

def run_script():
    with open('teams_players.json') as json_file:  
        players = json.load(json_file)

    data = pd.read_csv("../data/season1516processed.csv")
    real_outcome = get_outcome(data)

    elastic_model = pickle.load(open('elastic_models.pkl', 'rb'))
    lasso_model = pickle.load(open('lasso.pkl', 'rb'))
    ridge_model = pickle.load(open('ridge.pkl', 'rb'))
    gbtrees_model = pickle.load(open('gbtrees.pkl', 'rb'))

    nb_classifier = pickle.load(open('nb_classifier.pkl', 'rb'))
    svm_classifier = pickle.load(open('svm_classifier.pkl', 'rb'))

    print('=====================Elastic model=====================')
    predicted_elastic_nb = get_predictions(data, nb_classifier, elastic_model, players)
    predicted_elastic_svm = get_predictions(data, svm_classifier, elastic_model, players)
    print("Naive Bayes Accuracy Score -> ", f1_score(y_pred=predicted_elastic_nb, y_true=real_outcome)*100)
    print('SVM Accuracy Score -> ', f1_score(y_pred=predicted_elastic_svm, y_true=real_outcome)*100)

    print('=====================Lasso model=====================')
    predicted_lasso_nb = get_predictions(data, nb_classifier, lasso_model, players)
    predicted_lasso_svm = get_predictions(data, svm_classifier, lasso_model, players)
    print("Naive Bayes Accuracy Score -> ", f1_score(y_pred=predicted_lasso_nb, y_true=real_outcome)*100)
    print('SVM Accuracy Score -> ', f1_score(y_pred=predicted_lasso_svm, y_true=real_outcome)*100)

    print('=====================Ridge model=====================')
    predicted_ridge_nb = get_predictions(data, nb_classifier, ridge_model, players)
    predicted_ridge_svm = get_predictions(data, svm_classifier, ridge_model, players)
    print("Naive Bayes Accuracy Score -> ", f1_score(y_pred=predicted_ridge_nb, y_true=real_outcome)*100)
    print('SVM Accuracy Score -> ', f1_score(y_pred=predicted_ridge_svm, y_true=real_outcome)*100)

    print('=============Gradient boosted trees model=============')
    predicted_gbtrees_nb = get_predictions(data, nb_classifier, gbtrees_model, players)
    predicted_gbtrees_svm = get_predictions(data, svm_classifier, gbtrees_model, players)
    print("Naive Bayes Accuracy Score -> ", f1_score(y_pred=predicted_gbtrees_nb, y_true=real_outcome)*100)
    print('SVM Accuracy Score -> ', f1_score(y_pred=predicted_gbtrees_svm, y_true=real_outcome)*100)
    # datumski prvu utakmicu
    # l1 koji atributi uticu na krajnji rezultat
    # kombinacija metoda
    # majority voting, stacking


if __name__ == "__main__":
    run_script()
