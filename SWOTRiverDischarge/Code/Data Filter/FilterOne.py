import os
import glob
import netCDF4 as nc
import geopandas as gpd
import pandas  as pd
import datetime
workspace = r"C:\Users\lwx\Desktop\Dishcarge"
Files = glob.glob(os.path.join(workspace, "*.geojson"))
count = 10

widths = []
dateList = []

for File in Files:
    width = []
    gdf = gpd.read_file(File)
    print(len(gdf))
    indexList = [int(i * (len(gdf)/50)) for i in range(count)]
    for i in indexList:
        width.append(gdf.iloc[i]["width"])
    widths.append(width)
    dateList.append(File.split("\\")[-1].split(".")[0])
widthDf = pd.DataFrame(widths).transpose()
widthDf.columns = dateList
widthDf.to_csv(os.path.join(workspace,"width.csv"), index = False)
DischargeDf = pd.read_csv(os.path.join(workspace, "HanKou.txt"))

discharge = []
for date in dateList:
    start = datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(days=-2)
    end = datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(days=+2)
    gdf = DischargeDf[(DischargeDf["日期"] > start.strftime('%Y-%m-%d')) & (DischargeDf["日期"] < end.strftime('%Y-%m-%d'))]
    discharge.append(gdf["流量"].mean())

gdf2 = pd.DataFrame(discharge).transpose()
gdf2.columns = dateList
gdf2.to_csv(os.path.join(workspace, "discharge.csv"), index = False)