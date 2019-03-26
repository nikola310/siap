import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("../data/dataSet_processed.csv")
players = data.PLAYER_ID.unique()

# Split test and train dataset for each player
train = {}
test = {}
for val in players:
    is_player = data['PLAYER_ID'] == val
    subset = data[is_player]
    train_tmp, test_tmp = train_test_split(subset, test_size=0.3)
    train[val] = train_tmp
    test[val] = test_tmp

for val in players:
    est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
    est.fit(train[val]['DEFENSIVE_RATING'].values.reshape(-1, 1), train[val]['POINTS'])


errors=[] 
### Test
for val in players:    
    prediction = est.predict(test[val]['DEFENSIVE_RATING'].values.reshape(-1, 1))

    prediction = np.round(prediction,0)
    cnt=0
    compare={}
    for val1 in prediction:
        compare[val1]=test[val]['POINTS'].values[cnt]
        cnt+= 1
    print('----Predictions: real')
    print(compare)
    
    for val2 in compare:
        errors.append(abs(val2-compare[val2])/compare[val2])
    print(str(errors))
    err=np.mean(errors)
    #train_rdf_err = 1-(predict_rdf_train == train_target).mean()
    #err = 1 - (prediction == subset['POINTS']).mean()
    print('----Error is: ' + str(err*100) + '%')
    #print("Accuracy Score -> ", est.score(prediction.reshape(-1, 1), test[val]['POINTS']))