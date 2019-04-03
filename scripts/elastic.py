import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

data = pd.read_csv("../data/dataSet_processed.csv")
#train, test = train_test_split(data, test_size=0.3)
players = data.PLAYER_ID.unique()

train = {}
test = {}
for val in players:
    is_player = data['PLAYER_ID'] == val
    subset = data[is_player]
    train_tmp, test_tmp = train_test_split(subset, test_size=0.3)
    train[val] = train_tmp
    test[val] = test_tmp

finalError=[]
for val in players:

    errors2=[]

    ENreg = ElasticNet(alpha=0.5, l1_ratio=0.5, normalize=False)
    ENreg.fit(train[val]['DEFENSIVE_RATING'].values.reshape(-1, 1), train[val]['POINTS'])

    prediction = ENreg.predict(test[val]['DEFENSIVE_RATING'].values.reshape(-1, 1))
    #print(subset)
    prediction = np.round(prediction, 0)
    cnt=0
    compare={}
    for val1 in prediction:
        compare[val1] = test[val]['POINTS'].values[cnt]
        cnt+= 1

    for i in range(len(prediction)):
        errors2.append(abs(prediction[i] - test[val]['POINTS'].values[i]))

    err2 = np.mean(errors2)
    finalError.append(err2)
    #print('----Error for player ' + str(test[val]['PLAYER_NAME'].values[0]) + ' is in average ' + str(err2) + ' points')

print('----Average error for all players (lasso regression): ' + str(np.mean(finalError)) + ' points')
print('-------------------------------------------------------------------------------------------')
