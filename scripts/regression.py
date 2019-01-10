import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

data = pd.read_csv("../data/dataSet_processed.csv")

train, test = train_test_split(data, test_size=0.3)

#print(len(train))

ridgereg = Ridge(alpha=1.0, copy_X=True, normalize=True)

ridgereg.fit(train['DEFENSIVE_RATING'].values.reshape(-1, 1), train['POINTS'])
print(ridgereg)

prediction = ridgereg.predict(test['DEFENSIVE_RATING'].values.reshape(-1, 1))

print(prediction)