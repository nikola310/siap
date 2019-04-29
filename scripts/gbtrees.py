import pickle
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    data = pd.read_csv("../data/dataSet_processed.csv")
    players = data.PLAYER_NAME.unique()
    display_top_5 = True
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
        est = GradientBoostingRegressor(n_estimators=200, learning_rate=0.5,
                                        max_depth=0.5, random_state=0, loss='ls')
        est.fit(train[val]['DEFENSIVE_RATING'].values.reshape(-1, 1), train[val]['POINTS'])
        gbtrees_models[val] = est

    rms_error = []
    r_squared = []
    finalError = []
    f_error = []
    error_dict = {}

    for val in players:
        errors2 = []
        prediction = est.predict(test[val]['DEFENSIVE_RATING'].values.reshape(-1, 1))

        val_prediction = prediction
        val_actual = test[val]['POINTS'].values

        prediction = np.round(prediction, 0)
        cnt = 0
        compare = {}
        for i in range(len(prediction)):
            errors2.append(abs(prediction[i] - test[val]['POINTS'].values[i]))
            error_dict[val] = abs(prediction[i] - test[val]['POINTS'].values[i])

        err2 = np.mean(errors2)
        finalError.append(err2)
        #print('----RMS Error for player ' + str(test[val]['PLAYER_NAME'].values[0]) 
        #      + ' is ' + str(
        #rms_error.append(abs(sqrt(mean_squared_error(y_true=val_actual, y_pred=val_prediction))*100)) # + ' percent')
        #print('R Squared Error for player ' + str(test[val]['PLAYER_NAME'].values[0])
        #      + ' is ' + str
        #r_squared.append((r2_score(y_true=val_actual, y_pred=val_prediction)*100)) # + ' percent')    


    print('----Average error for all players (gbtrees): ' + str(np.round(np.mean(finalError))) + ' points')
    #print('----Average RMS error for all players (gbtrees): ' + str(np.mean(rms_error)) + ' points')
    #print('----Average R squared error for all players (gbtrees): ' + str(np.mean(r_squared)) + ' points')
    pickle.dump(gbtrees_models, open('gbtrees.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    if display_top_5:
        print('-------------------------------------------------------------------------------------------')
        df = pd.DataFrame(list(error_dict.items()), columns=['Player', 'Error'])
        max5 = df.nlargest(5, 'Error')
        max5.Player = max5.Player.str.title()
        ax = max5.plot("Player", "Error", kind="barh", legend=None, colormap='Purples_r')
        print(df.nsmallest(5, 'Error'))
        ax.set_ylabel("Igrač", fontsize=20)
        ax.set_xlabel('Odstupanje', fontsize=20)
        plt.show()