
import geopandas as gpd
import os
import netCDF4 as nc 
import pandas as pd

workspace = r"C:\Users\lwx\Desktop\Dishcarge"
station = os.path.join(workspace, "Hankou.shp")
stationLocationDF = gpd.read_file(station)

# COMID
basinDF = gpd.read_file(r"D:\data\GRADES\level_01_v0.7\pfaf_04_riv_3sMERIT.shp")
test = gpd.sjoin_nearest(stationLocationDF, basinDF, lsuffix='left', rsuffix='right')
ID = test["COMID"].values[0]
basinDF = None

# read prior
prior = nc.Dataset(r"D:\data\GRADES\output_pfaf_04_1979-2019.nc")
index = list(prior["rivid"]).index(ID)
Qmean = prior["Qout"][:,index].mean()

pd.DataFrame([Qmean]).to_csv(os.path.join(workspace, 'Qmean.csv'), index = False, header = False)