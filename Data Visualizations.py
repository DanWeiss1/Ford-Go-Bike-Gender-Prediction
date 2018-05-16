import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_pickle('rides.pkl')
weekdays = ["Mon.", "Tues.", "Weds.", "Thurs.", "Fri.", "Sat.", "Sun."]


def get_female_pct(df):
    return (df["member_gender"] == "Female").mean() * 100


# bar graph overall gender breakdown
plt.bar(['Male', 'Female'], [(df['member_gender'] == "Male").mean() * 100, \
                             (df['member_gender'] == "Female").mean() * 100])
plt.title("Ridership on Ford GoBike\n", fontsize=30, fontweight='bold')
plt.ylabel("Proportion of Total Rides", fontsize='18')
plt.xticks(fontsize='18')
plt.savefig('graphics/OverallRidership.png')

# bar graph of overall gender breakdown by day

dow = df["start_time"].apply(lambda x: x.dayofweek)
grouped = df.groupby(dow)
by_dow = grouped.apply(get_female_pct)
dowDF = pd.DataFrame({
    "Female Ridership": by_dow,
    "Day": weekdays
})
dowDF.set_index("Day")
ax = dowDF.plot(kind="bar", legend=None)
ax.set_xlabel("")
ax.set_title("Female Ridership on Ford GoBike\nby Day of Week\n",
             fontsize="30",
             fontweight="bold")
ax.set_ylabel("Trips Taken by Female Riders,\nas a Percentage of All Trips\n",
              fontsize="18")
ax.xaxis.grid(False)
ax.set_ylim(0, 30)
ax.set_yticklabels(["{0:.0f}%".format(y)
                    for y in ax.get_yticks()], fontsize="x-large")
ax.set_xticklabels(weekdays, rotation=45,
                   fontsize="18", fontweight="bold")

plt.savefig('graphics/FemaleRidership DOW.png')

# bar graph of overall gender breakdown by hour
hours = [i for i in range(24)]
hr = df["start_time"].apply(lambda x: x.hour)
grouped = df.groupby(hr)
by_hr = grouped.apply(get_female_pct)
hrDF = pd.DataFrame({
    "Female Ridership": by_hr,
})

ax = hrDF.plot(kind="bar", legend=None)
ax.set_xlabel("")
ax.set_title("Female Ridership on Ford GoBike\nby Hour of Day\n",
             fontsize="30",
             fontweight="bold")
ax.set_ylabel("Trips Taken by Female Riders,\nas a Percentage of All Trips\n",
              fontsize="18")
ax.xaxis.grid(False)
ax.set_ylim(0, 30)
ax.set_yticklabels(["{0:.0f}%".format(y)
                    for y in ax.get_yticks()], fontsize="x-large")
ax.set_xticklabels(hours, rotation=90,
                   fontsize="14", fontweight="bold")

plt.savefig('graphics/Trips by Hour of Day.png')

# plot histograms for men and women speed
mph_bins = [i for i in range(18)]
plt.title('Speed of Male and Female Rides\n', fontsize="30",
          fontweight="bold")
plt.hist(df[df['member_gender'] == 'Male']['mph'], label="Male", bins=mph_bins)
plt.hist(df[df['member_gender'] == 'Female']['mph'], label="Female", bins=mph_bins)
plt.xlabel('MPH', fontsize='18')
plt.ylabel("Number of Rides", fontsize='18')
plt.legend()
plt.show()

plt.savefig('graphics/Speed Histogram.png')
