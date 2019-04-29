import numpy as np
import pandas as pd
from pandas import Series
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    data = pd.read_csv("../data/shot_logs.csv")
    mpl.rcParams["font.size"] = 18
    made_missed = False
    home_away = False
    success_rate_home_away = True
    amount_home_away = False
    smt_home_away = False
    # SHOT_RESULT i FGM imaju iste vrednosti.
    # print(pd.get_dummies(data["SHOT_RESULT"])["made"].astype("bool").equals(data["FGM"].astype("bool")))

    # Balansirani odnosi postignutih i promasenih koseva
    if made_missed:
        plt.locator_params(axis='y', nbins=6)
        ax = data['FGM'].value_counts().plot(kind='bar', color=['#ea6e20', '#5d45e8'], width=0.35)

        ax.set_xticklabels(["Promašaj", "Pogodak"]) #, fontsize=15)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_xlabel("Šut")#, fontsize=20)
        ax.set_ylabel("Broj")#, fontsize=20)
        # ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
        # ax.yaxis.set_label_coords(-0.01, 0.5)
        #plt.tick_params(axis='y', which='major', labelsize=12)
        plt.show()
    elif home_away:
        # Balansirani sutevi kod kuce i u gostima
        fig, axarr = plt.subplots(1, 2, figsize=(24, 8))
        sns.countplot(x="LOCATION", data=data, ax=axarr[0])

        axarr[0].set_xticklabels(["Away", "Home"])
        axarr[0].set_xlabel("Game Location")
        axarr[0].set_ylabel("Total")
        location = pd.crosstab(data["LOCATION"], data["FGM"]).reset_index()
        location["Success_Rate"] = location[1] / (location[0] + location[1])
        plt.show()
    elif amount_home_away:
        made = 0
        missed = 0
        for row in data.iterrows():
            if row[1]['SHOT_RESULT'] == 'made':
                made += 1
            elif row[1]['SHOT_RESULT'] == 'missed':
                missed += 1

        ax = sns.countplot(x="LOCATION", data=data)
        ax.set_xticklabels(["U gostima", "Kod kuće"])
        ax.set_xlabel("Lokacija")
        ax.set_ylabel("Ukupno")
        plt.show()
    elif success_rate_home_away:
        location = pd.crosstab(data["LOCATION"], data["FGM"]).reset_index()
        location["Success_Rate"] = location[1] / (location[0] + location[1])

        ax = plt.bar(location["LOCATION"], location["Success_Rate"], color=['#009933', '#00ccff'], width=0.35, align='center')
        axes = plt.axes()
        axes.set_ylim([0, 0.5])
        axes.set_xticklabels(["U gostima", "Kod kuće"], fontsize=15)
        axes.set_xlabel("Lokacija", fontsize=20)
        axes.set_ylabel("Uspešnost", fontsize=20)
        # axes.set_yticklabels(axes.get_yticklabels(), fontsize=2)
        plt.show()

    elif smt_home_away:
        loc = pd.crosstab(data["LOCATION"], data["PTS_TYPE"]).reset_index()
        loc = pd.melt(loc, id_vars="LOCATION", var_name="point type", value_name="amount")
        print(loc)
        per = []
        for row in loc.iterrows():
            per.append(int(row[1]['amount']) / 128069)

        loc['percentage'] = per
        ax = sns.catplot(x='LOCATION', y='percentage', hue='point type', data=loc, kind='bar', palette=sns.color_palette("hls", 8))
        ax.axes[0,0].set_ylim((0, 0.5))
        ax.set_xticklabels(["U gostima", "Kod kuće"])
        ax.set_xlabels("Lokacija")
        ax._legend.set_title('Tip šuta')
        ax.set_ylabels("Procenat od ukupnih šuteva")
        plt.show()