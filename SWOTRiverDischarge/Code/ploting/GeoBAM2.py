import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import netCDF4 as nc

workspace = r"C:\Users\lwx\Desktop\Dishcarge"
plt.rc('font',family='Arial') 
# data
fileList = glob.glob(os.path.join(workspace, "*_geoBAM.txt"))
result = pd.read_csv(os.path.join(workspace, 'Brahmaputra_geoBAM.txt'))

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
ax = sns.lineplot(x="time", y="flow",
             hue="stat", style="series", alpha = 0.5, linewidth = 2,
             data=result)

trueData = nc.Dataset(os.path.join(r"C:\Users\lwx\Desktop\Dishcarge\Ideal-Data", "IowaRiver.nc"))
trueData = trueData["Reach_Timeseries/Q"][:].mean(axis = 1)
#trueData.index = pd.to_datetime(trueData.index)
#trueData.columns = ["true"]
ax.plot(trueData, linewidth = 3, color = "#333333", alpha = 0.7, label = "True")
plt.legend()
plt.savefig(os.path.join(workspace, "result.png"), dpi = 300)