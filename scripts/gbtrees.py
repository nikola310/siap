import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv("../data/dataSet_processed.csv")
players = data.PLAYER_NAME.unique()

# Split test and train dataset for each player
train = {}
test = {}
gbtrees_models = {}
for val in players:
    is_player = data['PLAYER_NAME'] == val
    subset = data[is_player]
    train_tmp, test_tmp = train_test_split(subset, test_size=0.3)
    train[val] = train_tmp
    test[val] = test_tmp

for val in players:
    est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
        max_depth=0.5, random_state=0, loss='ls')
    est.fit(train[val]['DEFENSIVE_RATING'].values.reshape(-1, 1), train[val]['POINTS'])
    gbtrees_models[val] = est


finalError = []
### Test
for val in players:
    errors2 = [] 
    prediction = est.predict(test[val]['DEFENSIVE_RATING'].values.reshape(-1, 1))

    prediction = np.round(prediction, 0)
    cnt = 0
    compare = {}
    for i in range(len(prediction)):
        errors2.append(abs(prediction[i] - test[val]['POINTS'].values[i]))

    err2 = np.mean(errors2)
    finalError.append(err2)
    #print('----Error for player ' + str(test[val]['PLAYER_NAME'].values[0]) + ' is in average ' + str(err2) + ' points')

print('----Average error for all players (gbtrees): ' + str(np.mean(finalError)) + ' points')  
print('-------------------------------------------------------------------------------------------')

pickle.dump(gbtrees_models, open('gbtrees.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
