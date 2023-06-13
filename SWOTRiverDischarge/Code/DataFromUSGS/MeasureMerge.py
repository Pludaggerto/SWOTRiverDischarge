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
import csv

class MeasureMerger(object):

    def __init__(self, workspace):

        logging.info("[INFO]Merging NWIS measurement...")
        self.workspace = workspace
        self.fileList = glob.glob(os.path.join(self.workspace, "measurement*"))

    def __del__(self):
        logging.info("[INFO]Merging NWIS measurement end...")

    def filter_data(self):
        for filename in self.fileList:
            with open(filename,'r') as f:
                lines=f.readlines()
                if lines[0][0] != '#':
                    continue
                while lines[0][0] == '#':
                    lines.pop(0)
                lines.pop(1)

            with open(filename,'w') as f: 
                for data in lines: 
                    f.write(data)
                f.flush()
        return

    def read_data(self):
        # !!!!!ERROR IN MEASUREMENT(23)
        df = pd.read_table(self.fileList[0], sep = "\t")
        for filename in self.fileList:
            try:
                df = df.append(pd.read_table(filename, sep = "\t"))
            except:
                logging.error("Error in " + filename)
        df.drop_duplicates(inplace = True)
        df = df[df["measured_rating_diff"] != "Poor"]
        df.to_csv(os.path.join(self.workspace, "mergeMeasurement.csv"), index = False)
        pass

    def run_all(self):
        self.filter_data()
        self.read_data()

def main():
    
    log = logging.getLogger()
    handler = logging.StreamHandler()
    log.addHandler(handler)
    log.setLevel(logging.INFO)

    workspace = r"D:\RiverDischargeData\USGSData"

    measureMerger = MeasureMerger(workspace)
    measureMerger.run_all()
     
if __name__ == '__main__':
    main() 
