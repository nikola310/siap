import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import partial_dependence

data = pd.read_csv("../data/shot_logs.csv")

data.drop(["CLOSEST_DEFENDER_PLAYER_ID"], axis=1, inplace=True)

match_info = data["MATCHUP"].str.split(" - ", n=1, expand=True)

data["MONTH"] = pd.to_datetime(match_info[0]).dt.month

teams = match_info[1].str.split(" ", expand=True)
#print(teams)

teams[0], teams[2] = np.where(teams[1] == "@", [teams[2], teams[0]], [teams[0], teams[2]])

#print(teams)

data["HOME_TEAM"] = teams[0]
data["AWAY_TEAM"] = teams[2]
data["TEAM"] = np.where(data["LOCATION"] == "H", data["HOME_TEAM"], data["AWAY_TEAM"])
data["OPPOSING_TEAM"] = np.where(data["LOCATION"] == "H", data["AWAY_TEAM"], data["HOME_TEAM"])
data.drop(["MATCHUP", "HOME_TEAM", "AWAY_TEAM"], axis=1, inplace=True)

#print(data)

period_time = pd.to_datetime(data["GAME_CLOCK"], format="%M:%S")
data.drop("GAME_CLOCK", axis=1, inplace=True)
data["PERIOD_CLOCK"] = period_time.dt.minute*60 + period_time.dt.second

#print(data)

#SHOT_RESULT i FGM imaju iste vrednosti.
print(pd.get_dummies(data["SHOT_RESULT"])["made"].astype("bool").equals(data["FGM"].astype("bool")))

data.drop("SHOT_RESULT", axis=1, inplace=True)

'''
    #Balansirani odnosi postignutih i promasenih koseva

ax = sns.countplot(x="FGM", data=data)
ax.set_xticklabels(["Missed", "Made"])
ax.set_xlabel("Shot")
ax.set_ylabel("Missed")

plt.show()
'''

'''
    #Balansirani sutevi kod kuce i u gostima
fig, axarr = plt.subplots(1, 2, figsize=(24,8))
sns.countplot(x="LOCATION", data=data, ax=axarr[0])
sns.countplot()
axarr[0].set_xticklabels(["Away", "Home"])
axarr[0].set_xlabel("Game Location")
axarr[0].set_ylabel("Total")
location = pd.crosstab(data["LOCATION"], data["FGM"]).reset_index()
location["Success_Rate"] = location[1] / (location[0] + location[1])
sns.barplot(location["LOCATION"], location["Success_Rate"], ax=axarr[1])
axarr[1].set_ylim((0, 1))
axarr[0].set_xticklabels(["Away", "Home"])
axarr[1].set_xlabel("Game Location")
axarr[1].set_ylabel("% of Shots Made")
fig.tight_layout()

plt.show()
'''

'''
    #Korelacija FINAL_MARGIN - FGM ?
fig, axarr = plt.subplots(1, 2, figsize=(24,8))
sns.distplot(data["FINAL_MARGIN"], kde=False, ax=axarr[0])
axarr[0].set_xlabel("Final Score Margin")
axarr[0].set_ylabel("Total")
final_margin = pd.crosstab(data["FINAL_MARGIN"], data["FGM"]).reset_index()
final_margin["Success_Rate"] = final_margin[1] / (final_margin[0] + final_margin[1])
sns.regplot(final_margin["FINAL_MARGIN"], final_margin["Success_Rate"], ax=axarr[1])
axarr[1].set_ylim((0, 1))
axarr[1].set_xlabel("Final Score Margin")
axarr[1].set_ylabel("% of Shots Made")
fig.tight_layout()

plt.show()
'''

'''
fig, axarr = plt.subplots(1, 2, figsize=(24, 8))
sns.countplot(x="SHOT_NUMBER", data=data, ax=axarr[0])
for label in axarr[0].xaxis.get_ticklabels()[::2]:
    label.set_visible(False)
axarr[0].set_xlabel("In-Game Shot Number")
axarr[0].set_ylabel("Total")
shot_number = pd.crosstab(data["SHOT_NUMBER"], data["FGM"]).reset_index()
shot_number["Success_Rate"] = shot_number[1] / (shot_number[0] + shot_number[1])
sns.regplot(shot_number["SHOT_NUMBER"], shot_number["Success_Rate"], ax=axarr[1])
axarr[1].set_ylim((0, 1))
axarr[1].set_xlabel("In-Game Shot Number")
axarr[1].set_ylabel("% of Shots Made")
fig.tight_layout()

plt.show()
'''

data.to_csv("../data/processed.csv")