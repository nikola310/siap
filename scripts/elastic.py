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
train, test = train_test_split(data, test_size=0.3)
players = train.PLAYER_ID.unique()


for val in players:

    is_player = train['PLAYER_ID'] == val
    subset = train[is_player]

    ENreg = ElasticNet(alpha=0.5, l1_ratio=0.5, normalize=False)
    ENreg.fit(subset['DEFENSIVE_RATING'].values.reshape(-1, 1), subset['POINTS'])
    is_player = test['PLAYER_ID'] == val
    subset = test[is_player]
    
    prediction = ENreg.predict(subset['DEFENSIVE_RATING'].values.reshape(-1, 1))
    print(subset)
    print('===================================================')
    print(prediction)

    #pred_cv = ENreg.predict(x_cv)

    #calculating mse

    #mse = np.mean((pred_cv - y_cv)**2)
    #print(mse)
    #ENreg.score(x_cv,y_cv)