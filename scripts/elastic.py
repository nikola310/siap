import pickle
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("../data/dataSet_processed.csv")
#train, test = train_test_split(data, test_size=0.3)
players = data.PLAYER_NAME.unique()

train = {}
test = {}
for val in players:
    is_player = data['PLAYER_NAME'] == val
    subset = data[is_player]
    train_tmp, test_tmp = train_test_split(subset, test_size=0.3)
    train[val] = train_tmp
    test[val] = test_tmp

elastic_models = {}
finalError = []
for val in players:

    errors2 = []

    ENreg = ElasticNet(alpha=0.5, l1_ratio=0.5, normalize=False)
    ENreg.fit(train[val]['DEFENSIVE_RATING'].values.reshape(-1, 1), train[val]['POINTS'])
    elastic_models[val] = ENreg

    prediction = ENreg.predict(test[val]['DEFENSIVE_RATING'].values.reshape(-1, 1))
    prediction = np.round(prediction, 0)

    val_prediction = prediction
    val_actual = test[val]['POINTS'].values

    cnt = 0
    compare = {}
    for val1 in prediction:
        compare[val1] = test[val]['POINTS'].values[cnt]
        cnt += 1

    for i in range(len(prediction)):
        errors2.append(abs(prediction[i] - test[val]['POINTS'].values[i]))

    err2 = np.mean(errors2)
    finalError.append(err2)
    '''
    print('----RMS Error for player ' + str(test[val]['PLAYER_NAME'].values[0]) 
          + ' is ' + str(sqrt(mean_squared_error(val_actual, val_prediction))*100) + ' percent')
    print('R Squared Error for player ' + str(test[val]['PLAYER_NAME'].values[0]) 
          + ' is ' + str(r2_score(val_actual, val_prediction)*100) + ' percent')
    '''
    #print('----Error for player ' + str(test[val]['PLAYER_NAME'].values[0]) + ' is in average ' + str(err2) + ' points')

print('----Average error for all players (lasso regression): ' + str(np.mean(finalError)) + ' points')
print('-------------------------------------------------------------------------------------------')

pickle.dump(elastic_models, open('elastic_models.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
