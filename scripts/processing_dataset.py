import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import partial_dependence

data = pd.read_csv("../data/dataSet.csv")

match_info = data["TEAMS"].str.split(" - ", n=1, expand=True)

data["MONTH"] = pd.to_datetime(match_info[0]).dt.month

teams = match_info[1].str.split(" ", expand=True)

teams[0], teams[2] = np.where(teams[1] == "@", [teams[2], teams[0]], [teams[0], teams[2]])

data["HOME_TEAM"] = teams[0]
data["AWAY_TEAM"] = teams[2]
data["TEAM"] = np.where(data["LOCATION"] == "H", data["HOME_TEAM"], data["AWAY_TEAM"])
data["OPPOSING_TEAM"] = np.where(data["LOCATION"] == "H", data["AWAY_TEAM"], data["HOME_TEAM"])
data.drop(["TEAMS", "HOME_TEAM", "AWAY_TEAM"], axis=1, inplace=True)
data.drop("GAME_ID", axis=1, inplace=True)
data.drop("LOCATION", axis=1, inplace=True)
data.drop("W/L", axis=1, inplace=True)
data.drop("FINAL_MARGIN", axis=1, inplace=True)
data.drop("MONTH", axis=1, inplace=True)

data.to_csv("../data/dataSet_processed.csv")