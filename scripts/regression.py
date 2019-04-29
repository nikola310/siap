import pickle
from math import sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
      data = pd.read_csv("../data/dataSet_processed.csv")
      test = {}
      players = data.PLAYER_NAME.unique()

      max5_lasso = False
      max5_ridge = True

      train = {}
      test = {}
      for val in players:
            is_player = data['PLAYER_NAME'] == val
            subset = data[is_player]
            train_tmp, test_tmp = train_test_split(subset, test_size=0.3)
            train[val] = train_tmp
            test[val] = test_tmp

      # Lasso regression
      lasso_models = {}
      for val in players:
            lassoreg = Lasso(tol=0.001, max_iter=1000, fit_intercept=False, normalize=True, alpha=0.95)

            lassoreg.fit(train[val]['DEFENSIVE_RATING'].values.reshape(-1, 1), train[val]['POINTS'])
            lasso_models[val] = lassoreg
      
      error_dict = {}
      finalError = []
      # Testing Lasso regression
      for val in players:
            errors2 = []
            prediction = lasso_models[val].predict(test[val]['DEFENSIVE_RATING'].values.reshape(-1, 1))
            prediction = np.round(prediction, 0)

            val_prediction = prediction
            val_actual = test[val]['POINTS'].values

            for i in range(len(prediction)):
                  errors2.append(abs(prediction[i] - test[val]['POINTS'].values[i]))
                  error_dict[val] = abs(prediction[i] - test[val]['POINTS'].values[i])
            #print(str(errors2))
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

      if max5_lasso:
            print('-------------------------------------------------------------------------------------------')
            df = pd.DataFrame(list(error_dict.items()), columns=['Player', 'Error'])
            max5 = df.nlargest(5, 'Error')
            max5.Player = max5.Player.str.title()
            ax = max5.plot("Player", "Error", kind="barh", legend=None, colormap='Purples_r')
            print(df.nsmallest(5, 'Error'))
            ax.set_ylabel("Igrač", fontsize=20)
            ax.set_xlabel('Odstupanje', fontsize=20)
            plt.show()


      # Ridge regression
      ridge_models = {}
      for val in players:
            ridgereg = Ridge(normalize=True, fit_intercept=False, alpha=0.95)
            ridgereg.fit(train[val]['DEFENSIVE_RATING'].values.reshape(-1, 1), train[val]['POINTS'])
            ridge_models[val] = ridgereg

      errors_ridge = {}
      # Testing Ridge regression
      finalError = []
      for val in players:  
            errors2 = []
            prediction = ridge_models[val].predict(test[val]['DEFENSIVE_RATING'].values.reshape(-1, 1))

            val_prediction = prediction
            val_actual = test[val]['POINTS'].values

            prediction = np.round(prediction, 0)

            for i in range(len(prediction)):
                  errors2.append(abs(prediction[i] - test[val]['POINTS'].values[i]))
                  errors_ridge[val] = abs(prediction[i] - test[val]['POINTS'].values[i])
                  

            err2 = np.mean(errors2)
            finalError.append(err2)
            '''
            print('----RMS Error for player ' + str(test[val]['PLAYER_NAME'].values[0])
                  + ' is ' + str(sqrt(mean_squared_error(val_actual, val_prediction))*100) + ' percent')
            print('R Squared Error for player ' + str(test[val]['PLAYER_NAME'].values[0])
                  + ' is ' + str(r2_score(val_actual, val_prediction)*100) + ' percent')
            '''
            #print('----Error for player ' + str(test[val]['PLAYER_NAME'].values[0]) + ' is in average ' + str(err2) + ' points')

      print('----Average error for all players (ridge regression): ' + str(np.mean(finalError)) + ' points')

      if max5_ridge:
            print('-------------------------------------------------------------------------------------------')
            dframe = pd.DataFrame(list(errors_ridge.items()), columns=['Player', 'Error'])
            max5 = dframe.nlargest(5, 'Error')
            max5.Player = max5.Player.str.title()
            ax = max5.plot("Player", "Error", kind="barh", legend=None, colormap='Purples_r')
            print(dframe.nsmallest(5, 'Error'))
            ax.set_ylabel("Igrač", fontsize=20)
            ax.set_xlabel('Odstupanje', fontsize=20)
            plt.show()

      pickle.dump(lasso_models, open('lasso.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
      pickle.dump(ridge_models, open('ridge.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)