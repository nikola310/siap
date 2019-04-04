import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split

np.random.seed(500)

data = pd.read_csv("../data/dataSet_processed.csv")
test = {}
#players = [203148, 202687, 2744, 203469, 202390, 201945, 202689, 203077]
players = data.PLAYER_NAME.unique() #data.PLAYER_ID.unique()
#print(len(data.PLAYER_ID.unique()))
#print(len(data.PLAYER_NAME.unique()))
# Split test and train dataset for each player
train = {}
test = {}
for val in players:
    is_player = data['PLAYER_NAME'] == val
    subset = data[is_player]
    train_tmp, test_tmp = train_test_split(subset, test_size=0.3)
    train[val] = train_tmp
    test[val] = test_tmp

### 1 - Lasso regression
lasso_models = {}
for val in players:

    ### Perform grid search to find best parameters
    #alphas = np.linspace(35.0, 0.001, 150).reshape(-1)
    #alphas = np.asarray(alphas)
    #normalize = ([True, False])

    lassoreg = Lasso(tol=0.001, max_iter=1000, fit_intercept=False, normalize=True, alpha=0.95)

    lassoreg.fit(train[val]['DEFENSIVE_RATING'].values.reshape(-1, 1), train[val]['POINTS'])
    lasso_models[val] = lassoreg
    #lasso_models[val] = grid.best_estimator_

finalError=[]
### Testing Lasso regression
for val in players:
    errors2 = []
    prediction = lasso_models[val].predict(test[val]['DEFENSIVE_RATING'].values.reshape(-1, 1))
    prediction = np.round(prediction,0)
    
    for i in range(len(prediction)):
        errors2.append(abs(prediction[i] - test[val]['POINTS'].values[i]))
    #print(str(errors2))
    err2 = np.mean(errors2)
    finalError.append(err2)
    #print('----Error for player ' + str(test[val]['PLAYER_NAME'].values[0]) + ' is in average ' + str(err2) + ' points')

print('----Average error for all players (lasso regression): ' + str(np.mean(finalError)) + ' points')  
print('-------------------------------------------------------------------------------------------')



### 2 - Ridge regression
ridge_models = {}
for val in players:

    ridgereg = Ridge(normalize=True, fit_intercept=False, alpha=0.95)
    
    ridgereg.fit(train[val]['DEFENSIVE_RATING'].values.reshape(-1, 1), train[val]['POINTS'])
    
    ridge_models[val] = ridgereg

    
### Testing Ridge regression
finalError=[]
for val in players:  
    errors2 = []
    prediction = ridge_models[val].predict(test[val]['DEFENSIVE_RATING'].values.reshape(-1, 1))
    prediction = np.round(prediction,0)
    
    for i in range(len(prediction)):
        #print(test[val]['POINTS'].values[i])
        errors2.append(abs(prediction[i] - test[val]['POINTS'].values[i]))
    #print(str(errors2))
    err2 = np.mean(errors2)
    finalError.append(err2)
    #print('----Error for player ' + str(test[val]['PLAYER_NAME'].values[0]) + ' is in average ' + str(err2) + ' points')

print('----Average error for all players (ridge regression): ' + str(np.mean(finalError)) + ' points')  
print('-------------------------------------------------------------------------------------------')

pickle.dump(lasso_models, open('lasso.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(ridge_models, open('ridge.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)