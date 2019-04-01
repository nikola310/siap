import numpy as np
import pandas as pd
import csv

names={}
defRating1516={}
names.update({'Detroit Pistons': 'DET','Cleveland Cavaliers': 'CLE','Atlanta Hawks': 'ATL','Chicago Bulls': 'CHI','New Orleans Pelicans': 'NOP','Boston Celtics': 'BOS','Philadelphia 76ers': 'PHI','Golden State Warriors': 'GSW','Brooklyn Nets': 'BKN','Utah Jazz': 'UTA','Denver Nuggets': 'DEN','Houston Rockets': 'HOU','Minnesota Timberwolves': 'MIN','Los Angeles Lakers': 'LAL','Memphis Grizzlies': 'MEM','Charlotte Hornets': 'CHA','Miami Heat': 'MIA','New York Knicks': 'NYK','San Antonio Spurs': 'SAS','Orlando Magic': 'ORL','Milwaukee Bucks': 'MIL','Oklahoma City Thunder': 'OKC','Washington Wizards': 'WAS','Dallas Mavericks': 'DAL','Phoenix Suns': 'PHX','Portland Trail Blazers': 'POR','Sacramento Kings': 'SAC','Indiana Pacers': 'IND','Toronto Raptors': 'TOR','Los Angeles Clippers': 'LAC'})
defRating1516.update({'SAS': 99.55,'ATL': 102,'IND': 103.51,'BOS': 104.1,'GSW': 104.32,'LAC': 104.57,'UTA': 104.78, 'CHA': 105.01,'CLE': 105.32,'MIA': 105.34,'TOR': 106.16,'DET': 106.19,'OKC': 106.22,'WAS': 106.48,'CHI': 107.2,'ORL': 107.47,'DAL': 107.48,'NYK': 108.43,'POR': 108.67,'MEM': 108.69,'SAC': 109.07,'HOU': 109.1,'DEN': 109.47,'MIL': 109.65,'PHX': 109.8,'PHI': 110.15,'NOP': 110.17,'MIN': 111.02,'BKN': 111.6,'LAL': 112.4})

octob = pd.read_csv("../data/VerificationDatasets/October.csv")
novem = pd.read_csv("../data/VerificationDatasets/November.csv")
decem = pd.read_csv("../data/VerificationDatasets/December.csv")
janua = pd.read_csv("../data/VerificationDatasets/January.csv")
febru = pd.read_csv("../data/VerificationDatasets/February.csv")
march = pd.read_csv("../data/VerificationDatasets/March.csv")
april = pd.read_csv("../data/VerificationDatasets/April.csv")
datasets=[octob,novem,decem,janua,febru,march,april]
data=pd.concat(datasets)
data.drop(data.columns[6], axis=1, inplace=True)
data.drop('Notes', axis=1, inplace=True)

processed=[]

for game in data.values:
    game=list(game)
    game[2]=names[game[2]]
    game[4]=names[game[4]]
    game.append(defRating1516[game[2]])
    game.append(defRating1516[game[4]])
    processed.append(game)
    print(game)

with open('../data/season1516processed.csv', 'w', newline='') as myfile:
    wr=csv.writer(myfile,quoting=csv.QUOTE_ALL)
    wr.writerow(['DATE','START','VISITOR','V_PTS','HOME','H_PTS','OT','ATTENDANCE','V_DEFRTG','H_DEFRTG'])
    for row in processed:
        wr.writerow(row)

