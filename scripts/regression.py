import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso

np.random.seed(500)

data = pd.read_csv("../data/dataSet_processed.csv")
test={}
players = [2548]
#players = data.PLAYER_ID.unique()

# Split test and train dataset for each player
train = {}
test = {}
for val in players:
    is_player = data['PLAYER_ID'] == val
    subset = data[is_player]
    train_tmp, test_tmp = train_test_split(subset, test_size=0.3)
    train[val] = train_tmp
    test[val] = test_tmp

### 1 - Lasso regression
lasso_models = {}
for val in players:

    ### Perform grid search to find best parameters
    alphas = np.linspace(35.0, 0.001, 150).reshape(-1)
    alphas = np.asarray(alphas)
    #normalize = ([True, False])

    lassoreg = Lasso(tol=0.001, max_iter=1000, fit_intercept=False)
    grid = GridSearchCV(estimator=lassoreg, param_grid=dict(alpha=alphas), cv=5, scoring='r2') #, normalize=normalize
    
    grid.fit(train[val]['DEFENSIVE_RATING'].values.reshape(-1, 1), train[val]['POINTS'])
    ### Summarize the results of the grid search
    print('Best score: ' + str(grid.best_score_))
    print('Alpha: ' + str(grid.best_estimator_.alpha))
    print('Fit intercept: ' + str(grid.best_estimator_.normalize))

    
    ## Just train with best parameters
    ##
    ##
    ##
    
    lasso_models[val] = grid.best_estimator_

errors=[] 
### Testing Lasso regression
for val in players:    
    prediction = lasso_models[val].predict(test[val]['DEFENSIVE_RATING'].values.reshape(-1, 1))

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
    
'''
### 2 - Ridge regression
ridge_models = {}
for val in players:

    is_player = train['PLAYER_ID'] == val
    subset = train[is_player]

    ### Perform grid search to find best parameters
    alphas = np.linspace(15.0, 0.0, 150).reshape(-1)
    alphas = np.asarray(alphas)
    fit_interceptOptions = ([True, False])
    solverOptions = (['svd', 'cholesky', 'sparse_cg', 'sag'])
    normalize = ([True, False])
    ridgereg = Ridge()
    grid = GridSearchCV(estimator=ridgereg, param_grid=dict(alpha=alphas, fit_intercept=fit_interceptOptions, solver=solverOptions, normalize=normalize), cv=5)
    
    grid.fit(subset['DEFENSIVE_RATING'].values.reshape(-1, 1), subset['POINTS'])
    ### Summarize the results of the grid search
    print('Best score: ' + str(grid.best_score_))
    print('Alpha: ' + str(grid.best_estimator_.alpha))
    print('Fit intercept: ' + str(grid.best_estimator_.fit_intercept))
    print('Solver: ' + str(grid.best_estimator_.solver))
    
    ## Just train with best parameters
    ##
    ##
    ##
    
    ridge_models[val] = grid.best_estimator_

    
### Testing Ridge regression
for val in players:
    is_player = test['PLAYER_ID'] == val
    subset = test[is_player]
    
    prediction = ridge_models[val].predict(subset['DEFENSIVE_RATING'].values.reshape(-1, 1))
'''