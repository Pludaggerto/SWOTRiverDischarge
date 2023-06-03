import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import time 
import datetime
import logging
import glob
import os 
import netCDF4 as nc
import shutil
import numpy as np
from tqdm import tqdm
from dbfread import DBF
import geopandas as gpd

class NHDGetter(object):

    def __init__(self, workspace):

        logging.info("[INFO]Getting NHD data begins...")
        self.workspace = workspace
        self.stationFile = os.path.join(self.workspace, "GageLoc.shp")
        self.recordFiles = []

    def __del__(self):
        logging.info("[INFO]Getting NHD data end...")

    def read_and_select_slopeDBF(self):
        for fileName in self.recordFiles:
            table = DBF(fileName, encoding='GBK')
            df = pd.DataFrame(iter(table))

    def get_all_dbf_file(self, folder):
        for fileName in glob.glob(os.path.join(folder, "*")):
            if os.path.isdir(fileName):
                self.get_all_dbf_file(fileName)
            if fileName.split("\\")[-1] == "elevslope.dbf":
                self.recordFiles.append(fileName)
        pass

    def read_station(self):
        gdf = gpd.read_file(self.stationFile)
        station_temp = gdf[["FLComID", "EVENTDATE", "SOURCE_FEA"]]
        station_temp = station_temp.set_index("SOURCE_FEA")
        station_temp.to_csv(os.path.join(self.workspace, "stationInfoNHD.txt"))

        NWIS_df = pd.read_table(r"C:\Users\lwx\source\repos\SWOTRiver\SWOTRiver\RiverDischarge\Code\DataFromUSGS\StationInfo.txt", sep = '\t')
        
        NWIS_df = NWIS_df.set_index("site_no")
        NWIS_df.index = NWIS_df.index.astype("str")

        mergeDF = NWIS_df.join(station_temp, how = "inner")
        mergeDF["site_no"] = mergeDF.index
        mergeDF.to_csv(os.path.join(self.workspace, "stationInfoMerge.txt"), index = False)
        return None

    def run_all(self):
        #self.get_all_dbf_file(self.workspace)
        #self.read_and_select_slopeDBF()
        self.read_station()

def main():
    
    log = logging.getLogger()
    handler = logging.StreamHandler()
    log.addHandler(handler)
    log.setLevel(logging.INFO)

    workspace = r"C:\Users\lwx\source\repos\SWOTRiver\SWOTRiver\RiverDischarge\Data\NHDSlope"

    NHDgetter = NHDGetter(workspace)
    NHDgetter.run_all()
     
if __name__ == '__main__':
    main() 