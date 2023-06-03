import dataretrieval.nwis as nwis
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

class USGSGetter(object):

    def __init__(self, workspace):

        logging.info("[INFO]Getting USGS data begins...")
        self.workspace = workspace
        df = pd.read_csv(os.path.join(self.workspace,"stationInfoMerge.txt"))
        self.stationInfo = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.dec_long_va, df.dec_lat_va))
        self.stationInfo_test = self.stationInfo.head()

    def __del__(self):
        logging.info("[INFO]Getting USGS data end...")

    def get_one_data(self, site, start, end):
        try:
            time.sleep(1)
            df = nwis.get_record(sites=str(site), service='measurements',)
            df.to_csv(os.path.join(self.workspace,"measurements.txt"), index = False, mode = "a+")
        except:
            logging.info("[INFO]ERROR in" + str(site)  + "...")

    def get_endDate(self, start):
        
        end = datetime.datetime.strptime(start, '%Y-%m-%d') + datetime.timedelta(days=+365)
        return end.strftime('%Y-%m-%d')

    def run_all(self):

        self.stationInfo = self.stationInfo[self.stationInfo["sv_count_nu"] > 20]
        i = 0
        for _, station in self.stationInfo.iterrows():
            i = i + 1
            try:
                logging.info("[INFO]trying " + str(i) + "/" + str(len(self.stationInfo))  + " ...")
                site  = station["site_no"]
                start = station["EVENTDATE"]
                if start != "--":
                    end = self.get_endDate(start)
                    self.get_one_data(site, start, end)
                else:
                    self.get_one_data(site, "2014-01-01", "2015-01-01")
            except:
                logging.info("[INFO]ERROR in ...")

def main():
    
    log = logging.getLogger()
    handler = logging.StreamHandler()
    log.addHandler(handler)
    log.setLevel(logging.INFO)

    workspace = os.path.dirname(__file__)

    USGSgetter = USGSGetter(workspace)
    USGSgetter.run_all()
     
if __name__ == '__main__':
    main() 
