import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

data = pd.read_csv("../data/dataSet_processed.csv")
data.drop("GAME_ID", axis=1, inplace=True)
data.drop("LOCATION", axis=1, inplace=True)
data.drop("W/L", axis=1, inplace=True)
data.drop("FINAL_MARGIN", axis=1, inplace=True)
data.drop("MONTH", axis=1, inplace=True)
#train, test = train_test_split(data, test_size=0.3)
players = data.PLAYER_ID.unique()


finalError=[]
for val in players:

    errors=[]
    is_player = data['PLAYER_ID'] == val
    subset = data[is_player]
    train, test = train_test_split(subset, test_size=0.3)

    ENreg = ElasticNet(alpha=0.5, l1_ratio=0.5, normalize=False)
    ENreg.fit(train['DEFENSIVE_RATING'].values.reshape(-1, 1), train['POINTS'])
    is_player = test['PLAYER_ID'] == val
    subset = test[is_player]
    
    prediction = ENreg.predict(subset['DEFENSIVE_RATING'].values.reshape(-1, 1))
    #print(subset)
    prediction = np.round(prediction,0)
    cnt=0
    compare={}
    for val1 in prediction:
        compare[val1]=subset['POINTS'].values[cnt]
        cnt+= 1
        
    print('----Predictions: real')
    print(compare)
    
    for val2 in compare:
        if compare[val2]!=0:
            errors.append(abs(val2-compare[val2])/compare[val2])
        else:
            errors.append(val2)
    print('Errors for each prediction: ')
    print(str(errors))
    err=np.mean(errors)
    finalError.append(err*100)
    print('----Average error for player is: ' + str(err*100) + '%')
    print('===================================================')


print('Final error for all: ' + str(np.mean(finalError)))
    #pred_cv = ENreg.predict(x_cv)

    #calculating mse

    #mse = np.mean((pred_cv - y_cv)**2)
    #print(mse)
    #ENreg.score(x_cv,y_cv)