import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

data = pd.read_csv("../data/dataSet_processed.csv")

train, test = train_test_split(data, test_size=0.3)

#print(len(train))

players = train.PLAYER_ID.unique()
models = {}

for val in players:
    ridgereg = Ridge(alpha=1.0, copy_X=True, normalize=True)
    is_player = train['PLAYER_ID'] == val
    subset = train[is_player]
    ridgereg.fit(subset['DEFENSIVE_RATING'].values.reshape(-1, 1), subset['POINTS'])
    models[val] = ridgereg

players_test = test.PLAYER_ID.unique()
res = {}


players_test = test.PLAYER_ID.unique()
res = {}

for val in players_test:
    is_player = test['PLAYER_ID'] == val
    subset = test[is_player]
    prediction = models[val].predict(subset['DEFENSIVE_RATING'].values.reshape(-1, 1))
    name = subset['PLAYER_NAME'].values[0]
    print('Player: ' + name)
    print(prediction)

print('===========================================================================')
print('Test set')
print(test)
'''
is_player = test['PLAYER_ID'] == 203148
subset = test[is_player]
prediction = models[val].predict(subset['DEFENSIVE_RATING'].values.reshape(-1, 1))
print(prediction)
print('----------------------------------------------------')
print(test[is_player])
'''
#print('{} {}'.format(test[is_player]['PLAYER_ID'], test[is_player]['POINTS']))
#print(test['PLAYER_ID'] + ' ' + test['POINTS'])

#ridgereg.fit(train['DEFENSIVE_RATING'].values.reshape(-1, 1), train['POINTS'])
#print(ridgereg)

#prediction = ridgereg.predict(test['DEFENSIVE_RATING'].values.reshape(-1, 1))

#print(prediction)