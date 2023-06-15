import pandas             as pd
import geopandas          as gpd

import logging
import os 

from Cluster import Clusterer
from MergeData import Merger
from WriteResult import ResultWriter

class Classification(Merger, Clusterer, ResultWriter):

    def __init__(self, workspace):
        super().__init__()
        logging.info("[INFO]Classification Experiment begins...")
        self.workspace = workspace

        # setting folder 
        self.USGSFolder = os.path.join(self.workspace, "USGSData")
        self.NHDFolder  = os.path.join(self.workspace, "NHDSlope")

        # NWIS 
        self.NWISFile   = os.path.join(self.USGSFolder, "mergeMeasurement.csv")
        self.NWISDf    = pd.read_csv(self.NWISFile)
        self.recordFiles = []

        # NHD
        self.NHDstationFile = os.path.join(self.NHDFolder, "GageLoc.shp")
        self.NHDstationGdf = gpd.read_file(self.NHDstationFile)
        self.NHDLakeFile = os.path.join(self.NHDFolder, "NHDLake.csv")
        self.sinuosityFile = os.path.join(self.NHDFolder, "Sinuousity_CONUS.TXT")

        # mergeFile
        self.mergedFolder   = os.path.join(self.workspace, 'USGS_NHD_Data')
        self.mergedDataFile = os.path.join(self.mergedFolder, "USGS_NHD_origin.csv")
        self.mergedCaledDataFile = os.path.join(self.mergedFolder, "USGS_NHD_caled.csv")
        self.FeatureFile = os.path.join(self.mergedFolder, "Classification.csv")

        # PCA
        self.PCAFolder = self.set_folder(self.mergedFolder, "PCA")

        # DBSCAN
        self.DBSCANFolder = self.set_folder(self.mergedFolder, "DBSCAN")

        # AffinityPropagation
        self.AffinityPropagationFolder = self.set_folder(self.mergedFolder, "AffPro")

        # mean shift
        self.meanShiftFolder = self.set_folder(self.mergedFolder, "meanShift")

        # cluster
        self.OPTICSFolder = self.set_folder(self.mergedFolder, "OPTICS")
        
        # BIRCH
        self.BIRCHFolder = self.set_folder(self.mergedFolder, "BIRCH")

    def __del__(self):
        logging.info("[INFO]Classification Experiment end...")

    def run_all(self):

        # mergeData
        if not os.path.exists(self.mergedDataFile):
            self.merge_NWIS_NHD()

        if not os.path.exists(self.NHDLakeFile):
            self.read_NHD_lake()

        if not os.path.exists(self.FeatureFile):
            self.select_and_cal_valuable()
        else:
            self.mergedData = pd.read_csv(self.mergedCaledDataFile)
            self.dataDF = pd.read_csv(self.FeatureFile)       
        
        # cluster
        self.PCA_cluster()
        self.DBSCAN_cluster()
        self.AffinityPropagation_cluster()
        self.meanShift_cluster()
        self.OPTICS_cluster()
        self.BIRCH_cluster()

def main():
    
    log = logging.getLogger()
    handler = logging.StreamHandler()
    log.addHandler(handler)
    log.setLevel(logging.INFO)

    workspace = r"D:\RiverDischargeData"

    classification = Classification(workspace)
    classification.run_all()
     
if __name__ == '__main__':
    main() 
