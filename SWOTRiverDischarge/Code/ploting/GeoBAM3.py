import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import netCDF4 as nc

workspace = r"C:\Users\lwx\Desktop"
plt.rc('font',family='Arial') 

name = "SeineDownstream"
# data
data   = pd.read_csv(os.path.join(workspace, name + '.txt'))
data_w = pd.read_csv(os.path.join(workspace, name + '_w.txt'))

def time2date(string):
    trueData = pd.read_csv(os.path.join(workspace, "discharge.csv"))
    dateList = list(trueData.columns[0:5]) + list(trueData.columns[6:])
    dateDict = {}
    for i in range(len(dateList)):
        dateDict[i+1] = dateList[i]
    return dateDict[string] 

# true 
# result["date"] = result["time"].apply(time2date)
# result["date"] = pd.to_datetime(result["date"])

data["flow($m^3/s$)"] = data["flow"]
data = data[data["stat"] == "mean"]

data_w["flow($m^3/s$)"] = data_w["flow"]
data_w = data_w[data_w["stat"] == "mean"]

data = data.groupby("time").mean()
data_w = data_w.groupby("time").mean()

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
ax.plot(data.index, data["flow($m^3/s$)"], alpha = 0.5, linewidth = 2, label = "W,S,dA")
               
ax.plot(data_w.index, data_w["flow($m^3/s$)"], alpha = 0.5, linewidth = 2, label = "only W")
          
trueData = nc.Dataset(os.path.join(r"D:\Desktop\Dishcarge\Ideal-Data", name + ".nc"))
trueData = trueData["Reach_Timeseries/Q"][:].mean(axis = 1)
#trueData.index = pd.to_datetime(trueData.index)
#trueData.columns = ["true"]
ax.plot(trueData, linewidth = 3, color = "#333333", alpha = 0.7, label = "True")
plt.legend()
plt.savefig(os.path.join(workspace, name + ".png"), dpi = 300)