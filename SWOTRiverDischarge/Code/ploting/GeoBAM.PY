import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

workspace = r"C:\Users\lwx\Desktop\Dishcarge"
plt.rc('font',family='Arial') 
# data
fileList = glob.glob(os.path.join(workspace, "Hankou_geoBAM.txt"))
result = pd.read_csv(fileList[0])

def time2date(string):
    trueData = pd.read_csv(os.path.join(workspace, "discharge.csv"))
    dateList = list(trueData.columns[0:5]) + list(trueData.columns[6:])
    dateDict = {}
    for i in range(len(dateList)):
        dateDict[i+1] = dateList[i]
    return dateDict[string] 

# true 
result["date"] = result["time"].apply(time2date)
result["date"] = pd.to_datetime(result["date"])
draw_highlow = True
if draw_highlow:
    result = result[result["stat"] == "mean"]
ax = sns.lineplot(x="date", y="flow",
             hue="stat", style="series", alpha = 0.5, linewidth = 1,
             data=result)

trueData = pd.read_csv(os.path.join(workspace, "discharge.csv")).transpose()
trueData = trueData[0:5].append(trueData[6:])
trueData.index = pd.to_datetime(trueData.index)
trueData.columns = ["true"]
ax.plot(trueData, linewidth = 2, color = "#333333", alpha = 0.7, label = "true")
plt.legend()
plt.savefig(os.path.join(workspace, "result.png"), dpi = 300)