import dataretrieval.nwis as nwis
import pandas             as pd
import geopandas          as gpd
import matplotlib.pyplot  as plt
import netCDF4            as nc
import numpy              as np
import scipy.stats.mstats as sp
import seaborn            as sns
import statsmodels.api    as sm

import time 
import datetime
import logging
import glob
import os 
import shutil
import csv
import math

from math            import sqrt
from ast             import literal_eval
from tqdm            import tqdm
from dbfread         import DBF
from pandas.plotting import scatter_matrix

from sklearn                 import metrics
from sklearn                 import svm
from sklearn                 import linear_model
from sklearn.cluster         import DBSCAN, KMeans
from sklearn.preprocessing   import StandardScaler, normalize
from sklearn.decomposition   import PCA
from sklearn.metrics         import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors       import NearestNeighbors

class DatasetCreater(object):

    def __init__(self, workspace):

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

    def __del__(self):
        logging.info("[INFO]Classification Experiment end...")

    #AHG exp function
    def regress(self, data, yvar, xvars):

        Y = np.log(data[yvar])
        X = np.log(data[xvars])
        X['intercept'] = 1.
        result = sm.OLS(Y, X).fit()
        return result.params[0] #only get exp

    #AHG int function
    def regress2(self, data, yvar, xvars):

        Y = np.log(data[yvar])
        X = np.log(data[xvars])
        X['intercept'] = 1.
        result = sm.OLS(Y, X).fit()
        return result.params[1] #only get int

    #bankfull hydraulics function
    def calculate_bankful(self, df, colname, retPeriod):

        # sort data smallest to largest
        sorted_data = df.sort_values(by=colname, ascending = False)
        # count total obervations
        n = sorted_data.shape[0]
        # add a numbered column 1 -> n to use in return calculation for rank
        sorted_data.insert(0, 'rank', range(1, 1 + n))
        #find desired rank
        desiredRank = (n+1)/retPeriod
        desiredRank = round(desiredRank)
        #get variable with desired rank
        output = sorted_data.loc[sorted_data['rank'] == desiredRank, colname]
        return(output)

    def set_folder(self, folder, name):
        """get path, if not exist, create it."""
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    #AHG r2 function
    def regress3(self, data, yvar, xvars):

        Y = np.log(data[yvar])
        X = np.log(data[xvars])
        X['intercept'] = 1.
        result = sm.OLS(Y, X).fit()
        return result.rsquared #only get r2

    def get_all_specific_file(self, folder, name, List):

        for fileName in glob.glob(os.path.join(folder, "*")):
            if os.path.isdir(fileName):
                self.get_all_specific_file(fileName, name, List)
            if fileName.split("\\")[-1] == name:
                List.append(fileName)
        return 

    def read_one_file(self, folder, name, List, columns):

        self.get_all_specific_file(folder, name, List)
        table = DBF(List[0], encoding='GBK')
        df = pd.DataFrame(iter(table))
        for i in range(1, len(List)):
            table = DBF(List[i], encoding='GBK')
            df = df.append(pd.DataFrame(iter(table)))
        df = df[~df[columns[0]].isna()]
        df[columns[0]] = df[columns[0]].astype(int).astype(str)
        df = df[columns]
        return df

    def read_one_shapefile(self, folder, name, List, columns):

        self.get_all_specific_file(folder, name, List)
        gdf = gpd.read_file(List[0])
        for i in range(1, len(List)):
            gdf = gdf.append(gpd.read_file(List[i]))
        gdf = gdf[~gdf[columns[0]].isna()]
        gdf[columns[0]] = gdf[columns[0]].astype(int).astype(str)
        gdf = gdf[columns]
        return gdf

    #bankfull hydraulics function
    def calculate_bankful(self, df, colname, retPeriod):

        # sort data smallest to largest
        sorted_data = df.sort_values(by=colname, ascending = False)
        # count total obervations
        n = sorted_data.shape[0]
        # add a numbered column 1 -> n to use in return calculation for rank
        sorted_data.insert(0, 'rank', range(1, 1 + n))
        #find desired rank
        desiredRank = (n+1)/retPeriod
        desiredRank = round(desiredRank)
        #get variable with desired rank
        output = sorted_data.loc[sorted_data['rank'] == desiredRank, colname]
        return(output)

    def read_and_select_NHD(self):

        # NHDFlowline
        self.NHDFlowlineDBFList = []
        NHDFlowlineDF = self.read_one_shapefile(self.NHDFolder, "NHDFlowline.shp", self.NHDFlowlineDBFList, ["COMID", "FCODE", "FDATE", "FTYPE" , "WBAREACOMI"])
        NHDFlowlineDF.columns = ["COMID", "FCODE", "FDATE", "FTYPE", "WBAREACOMI"]

        # elevslope        
        self.slopeDBFList = []
        slopeDF = self.read_one_file(self.NHDFolder, "elevslope.dbf", self.slopeDBFList, ["COMID", "SLOPE"])
        slopeDF.columns = ["COMID", "SLOPE"]

        # PlusFlowlineVAA
        self.PlusFlowlineVAADBFList = []
        PlusFlowlineVAADF = self.read_one_file(self.NHDFolder, "PlusFlowlineVAA.dbf", self.PlusFlowlineVAADBFList, ["ComID", "StreamOrde", "ArbolateSu", "LengthKM", "Tidal", "TOTMA"])
        PlusFlowlineVAADF.columns = ["COMID", "StreamOrde", "ArbolateSu", "LengthKM", "Tidal", "TOTMA"]

        # GageInfo
        table = DBF(os.path.join(self.NHDFolder, "GageInfo.dbf"), encoding='GBK')
        GageInfoDF = pd.DataFrame(iter(table))
        GageInfoDF = GageInfoDF[["GAGEID", "LatSite", "LonSite", "DASqKm"]]
        GageInfoDF.columns = ["site_no", "LatSite", "LonSite", "DASqKm"]

        station = self.NHDstationGdf[["FLComID", "SOURCE_FEA"]]
        station["COMID"] = station["FLComID"].astype(str)
        GageInfoDF["site_no"] = GageInfoDF["site_no"].astype(str)
        GageInfoDFID = GageInfoDF.merge(station, how='left', left_on="site_no", right_on="SOURCE_FEA")

        # join
        mergeDF = NHDFlowlineDF.merge(slopeDF, left_on = "COMID", right_on="COMID", how = "inner")
        mergeDF = mergeDF.merge(PlusFlowlineVAADF, left_on = "COMID", right_on="COMID", how = "inner")
        mergeDF = mergeDF.merge(GageInfoDFID, left_on = "COMID", right_on="COMID", how = "inner")

        mergeDF.to_csv(os.path.join(self.NHDFolder, "MergedNHDALL.csv"), index = False)

    def read_NHD_lake(self):
        
        self.LakeList = []
        LakeDF = self.read_one_file(self.NHDFolder, "PlusWaterbodyLakeMorphology.dbf", self.LakeList, ["ComID"])
        LakeDF.columns = ["COMID"]
        LakeDF.to_csv(self.NHDLakeFile)

    def get_merge_data(self):
        
        station = self.NHDstationGdf[["FLComID", "SOURCE_FEA","Measure"]]
        station["FLComID"] = station["FLComID"].astype(str)

        self.NWISDf["site_no"] = self.NWISDf["site_no"].astype(str)
        
        self.NHDDf = pd.read_csv(os.path.join(self.NHDFolder, "MergedNHDALL.csv"))
        self.NHDDf = self.NHDDf[~self.NHDDf["COMID"].isnull()]
        self.NHDDf["COMID"] = self.NHDDf["COMID"].astype(int).astype(str)
        
        mergePre = self.NWISDf.merge(station, how='left', left_on="site_no", right_on="SOURCE_FEA")
        mergeNHD = mergePre.merge(self.NHDDf, how='inner', left_on="FLComID", right_on="COMID",suffixes = ["", "_r"])
        mergeNHD.to_csv(self.mergedDataFile, index = False)
        return None

    def merge_NWIS_NHD(self):
        
        self.read_and_select_NHD()
        self.get_merge_data()
        return 

    def select_and_cal_valuable(self):

        self.med_lake = pd.read_csv(self.NHDLakeFile)
        self.mergedData = pd.read_csv(self.mergedDataFile)

        # get depth:
        self.mergedData['chan_depth'] = self.mergedData['chan_area'] / self.mergedData['chan_width']
        
        # sinuosity : Deviation from a path of maximum downslope
        sinuosityCONUS = pd.read_csv(self.sinuosityFile)
        self.mergedData = pd.merge(self.mergedData, sinuosityCONUS, left_on='COMID', right_on='COMID')

        # cleaning NA values and hydraulics below 0 and minimum 20 measurements
        self.mergedData = self.mergedData[self.mergedData['chan_width'] > 0]
        self.mergedData = self.mergedData[self.mergedData['chan_depth'] > 0]
        self.mergedData = self.mergedData[self.mergedData['chan_discharge'] > 0]
        self.mergedData = self.mergedData[self.mergedData['measured_rating_diff'] != 'Poor']
        self.mergedData = self.mergedData[self.mergedData['measured_rating_diff'] != 'POOR']
        self.mergedData = self.mergedData[self.mergedData['chan_width'].notnull()]
        self.mergedData = self.mergedData[self.mergedData['chan_depth'].notnull()]
        self.mergedData = self.mergedData[self.mergedData['chan_discharge'].notnull()]
        self.mergedData = self.mergedData[self.mergedData['chan_width'].notna()]
        self.mergedData = self.mergedData[self.mergedData['chan_depth'].notna()]
        self.mergedData = self.mergedData[self.mergedData['chan_discharge'].notna()]

        #convert needed units to metric
        self.mergedData['chan_width'] = self.mergedData['chan_width']*0.305
        self.mergedData['chan_depth'] = self.mergedData['chan_depth']*0.305
        self.mergedData['chan_velocity'] = self.mergedData['chan_velocity']*0.305 
        self.mergedData['chan_discharge'] = self.mergedData['chan_discharge']*0.028

        self.mergedData['n'] = ((self.mergedData['chan_depth'])**(2/3)*self.mergedData['SLOPE']**(1/2))/self.mergedData['chan_velocity']
        # 
        self.mergedData['shearStress'] = 9.81*self.mergedData['chan_depth']*self.mergedData['SLOPE']
        # Froude number
        self.mergedData['Fb'] = self.mergedData['chan_velocity']/((self.mergedData['chan_depth']*9.81)**(1/2))
        self.mergedData['minEntrain'] = 11*self.mergedData['chan_depth']*self.mergedData['SLOPE']

        #A0- median not minimum
        self.mergedData = self.mergedData.join(self.mergedData.groupby('site_no')['chan_area'].agg(['median']), on='site_no')
        self.mergedData = self.mergedData.rename(columns={"median": "A0"})
        bank_width = self.mergedData.groupby('site_no').apply(self.calculate_bankful, 'chan_width', 2).to_frame()
        bank_width = bank_width.rename(columns={'chan_width':'bank_width'})
        self.mergedData =  pd.merge(self.mergedData, bank_width, on='site_no')

        bank_depth = self.mergedData.groupby('site_no').apply(self.calculate_bankful, 'chan_depth', 2).to_frame()
        bank_depth = bank_depth.rename(columns={'chan_depth':'bank_depth'})
        self.mergedData =  pd.merge(self.mergedData, bank_depth, on='site_no')

        bank_Q = self.mergedData.groupby('site_no').apply(self.calculate_bankful, 'chan_discharge', 2).to_frame()
        bank_Q = bank_Q.rename(columns={'chan_discharge':'bank_Q'})
        self.mergedData =  pd.merge(self.mergedData, bank_Q, on='site_no')
        #AHG parameters
        b_temp = self.mergedData.groupby('site_no').apply(self.regress, 'chan_width', ['chan_discharge']).to_frame()
        a_temp = self.mergedData.groupby('site_no').apply(self.regress2, 'chan_width', ['chan_discharge']).to_frame()
        f_temp = self.mergedData.groupby('site_no').apply(self.regress, 'chan_depth', ['chan_discharge']).to_frame()
        c_temp = self.mergedData.groupby('site_no').apply(self.regress2, 'chan_depth', ['chan_discharge']).to_frame()
        m_temp = self.mergedData.groupby('site_no').apply(self.regress, 'chan_velocity', ['chan_discharge']).to_frame()
        k_temp = self.mergedData.groupby('site_no').apply(self.regress2, 'chan_velocity', ['chan_discharge']).to_frame()

        b_temp = b_temp.rename(columns={0:'b'})
        self.mergedData =  pd.merge(self.mergedData, b_temp, on='site_no')

        a_temp = a_temp.rename(columns={0:'loga'})
        self.mergedData =  pd.merge(self.mergedData, a_temp, on='site_no')

        c_temp = c_temp.rename(columns={0:'logc'})
        self.mergedData =  pd.merge(self.mergedData, c_temp, on='site_no')

        f_temp = f_temp.rename(columns={0:'f'})
        self.mergedData =  pd.merge(self.mergedData, f_temp, on='site_no')

        k_temp = k_temp.rename(columns={0:'logk'})
        self.mergedData =  pd.merge(self.mergedData, k_temp, on='site_no')

        m_temp = m_temp.rename(columns={0:'m'})
        self.mergedData =  pd.merge(self.mergedData, m_temp, on='site_no')

        #Calculate some more variables
        self.mergedData['r'] = self.mergedData['f']/self.mergedData['b']
        self.mergedData['unitPower'] = (998*9.8*self.mergedData['chan_discharge']*self.mergedData['SLOPE'])/self.mergedData['chan_width']
        self.mergedData['DistDwnstrm'] = self.mergedData['ArbolateSu']-((self.mergedData['Measure']/100)*self.mergedData['LengthKM'])
        self.mergedData['chan_material'] = np.where(self.mergedData['chan_material'] == 'silt', 'SILT', self.mergedData['chan_material'])
        self.mergedData['chan_material_index'] = np.where(self.mergedData['chan_material'] == 'BLDR', 1,
                                                        np.where(self.mergedData['chan_material'] == 'GRVL', 2,
                                                                np.where(self.mergedData['chan_material'] == 'SAND', 3,
                                                                        np.where(self.mergedData['chan_material'] == 'SILT', 4,
                                                                                np.where(self.mergedData['chan_material'] == 'UNSP', 5,5)))))

        self.mergedData['FCODEnorm'] = np.where(self.mergedData['FCODE'] == 33400, 1, #connectors or canals
                                                        np.where(self.mergedData['FCODE'] == 33600, 1, #connectors or canal
                                                                np.where(self.mergedData['FCODE'] == 46003, 2, #intermittent river
                                                                        np.where(self.mergedData['FCODE'] == 46006, 3, #perienial river
                                                                                np.where(self.mergedData['WBAREACOMI'].isin(self.med_lake['COMID']), 4,3))))) #lake if also in lakes dataset, otherwise its a main stem river or tidal reach and can be reclassified as perrenial river (basically....)

        self.mergedData['FTYPE'] = np.where(self.mergedData['FCODE'] == 33400, 'ArtificalChannel', #connector or canal
                                                        np.where(self.mergedData['FCODE'] == 33600, 'ArtificalChannel', #connector or canal
                                                                np.where(self.mergedData['FCODE'] == 46003, 'IntermittentRiver', #intermittent river
                                                                        np.where(self.mergedData['FCODE'] == 46006, 'PerennialRiver', #perienial river
                                                                                np.where(self.mergedData['WBAREACOMI'].isin(self.med_lake['COMID']), 'Lake/Reservoir/Wetland','PerennialRiver')))))

        self.mergedData = self.mergedData[self.mergedData['b'] > 0]
        self.mergedData = self.mergedData[self.mergedData['b'] < 1]
        self.mergedData = self.mergedData[self.mergedData['f'] > 0]
        self.mergedData = self.mergedData[self.mergedData['f'] < 1]
        self.mergedData = self.mergedData[self.mergedData['m'] > 0]
        self.mergedData = self.mergedData[self.mergedData['m'] < 1]

        groupSize = self.mergedData.groupby('site_no').size().to_frame()
        groupSize = groupSize.rename(columns={0:'groupSize'})
        self.mergedData = self.mergedData.merge(groupSize, on='site_no')

        self.mergedData = self.mergedData[self.mergedData['groupSize'] >= 20]
        self.mergedData['logA0'] = np.log(self.mergedData['A0'])
        self.mergedData['logr'] = np.log(self.mergedData['r'])
        self.mergedData['logn'] = np.log(self.mergedData['n'])
        self.mergedData['logWb'] = np.log(self.mergedData['bank_width'])
        self.mergedData['logDb'] = np.log(self.mergedData['bank_depth'])
        self.mergedData['logQb'] = np.log(self.mergedData['bank_Q'])
        self.mergedData['logQ'] = np.log(self.mergedData['chan_discharge'])

        print('\033[1m' + "# measurements:")
        print(len(self.mergedData.index))

        print('\033[1m' + "# cross-sections:")

        print(self.mergedData.groupby('site_no').ngroups)

        print('\033[1m' + "# rivers:")
        #display(self.mergedData.groupby('river_name').ngroups)
        self.mergedData.to_csv(self.mergedCaledDataFile)

        station_medians = self.mergedData.groupby('site_no').median() #get average hydraulics for each station
        station_medians = station_medians[['chan_width', 'n', 'SLOPE', 'StreamOrde','DistDwnstrm', 'FCODEnorm', 'chan_depth', 'chan_velocity', 'unitPower', 'r', 'DASqKm', 'Fb', 'shearStress', 'minEntrain', 'TOTMA', 'sinuosity', 'logA0', 'logn', 'b', 'logr', 'logDb', 'logWb']]#, 'CAT_SILTAVE', 'CAT_SANDAVE', 'CAT_CLAYAVE']]
        station_var = self.mergedData.groupby('site_no').var() #get average hydraulics for each station
        station_var = station_var[['chan_velocity', 'chan_depth', 'chan_width', 'n', 'unitPower', 'Fb', 'shearStress', 'minEntrain']]

        station_var.columns = ['velocity_var','depth_var','width_var',
                             'n_var','unitPower_var','Fb_var',
                              'shearStress_var', 'minEntrain_var']

        dataDF = station_medians.merge(station_var, on='site_no')
        
        dataDF['logchan_width'] = np.log(dataDF['chan_width'])

        self.dataDF = dataDF.dropna()
        self.dataDF.to_csv(self.)




    def DBSCAN_classfication(self):
        dataDF = self.dataDF
        features = self.dataDF[['chan_width', 'n', 'SLOPE', 'StreamOrde','DistDwnstrm', 'FCODEnorm', 'chan_depth', 'chan_velocity', 
            'unitPower', 'r', 'DASqKm', 'Fb', 'shearStress', 'minEntrain', 'TOTMA', 'sinuosity', 'velocity_var',
            'depth_var','width_var', 'n_var','unitPower_var','Fb_var','shearStress_var', 'minEntrain_var']]
        scaler = StandardScaler() 
        features = features.dropna()
        X_scaled = scaler.fit_transform(features) 

        # Normalizing the data so that the data 
        # approximately follows a Gaussian distribution 
        X_normalized = normalize(X_scaled) 
  
        # Converting the numpy array into a pandas DataFrame 
        X_normalized = pd.DataFrame(X_normalized) 
  
        # Renaming the columns 
        X_normalized.columns = features.columns 
        self.X_normalized = X_normalized

        clustering = DBSCAN(eps = 0.55, min_samples = 5).fit(self.X_normalized)
        labels = clustering.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)

        sns.set_style("white")
        dataDF
        fig, axs = plt.subplots(ncols=7, figsize=(20, 7))
        sns.boxplot(x="cluster", y='logA0', data=self.dataDF, palette='deep', ax=axs[0])
        sns.boxplot(x="cluster", y='logn', data=self.dataDF, palette='deep', ax=axs[1])
        sns.boxplot(x="cluster", y='b', data=self.dataDF, palette='deep', ax=axs[2])
        sns.boxplot(x="cluster", y='logWb', data=self.dataDF, palette='deep', ax=axs[3])
        sns.boxplot(x="cluster", y='logDb', data=self.dataDF, palette='deep', ax=axs[4])
        sns.boxplot(x="cluster", y='logr', data=self.dataDF, palette='deep', ax=axs[5])
        sns.boxplot(x="cluster", y='logchan_width', data=self.dataDF, palette='deep', ax=axs[6])

        fig.autofmt_xdate(rotation=90)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        axs[0].set_ylabel('log A0', fontsize=15)
        axs[1].set_ylabel('log Mannings n', fontsize=15)
        axs[2].set_ylabel('b', fontsize=15)
        axs[3].set_ylabel('log Wb', fontsize=15)
        axs[4].set_ylabel('log Db', fontsize=15)
        axs[5].set_ylabel('log r', fontsize=15)
        axs[6].set_ylabel('log W', fontsize=15)

        axs[0].set_xlabel('')
        axs[1].set_xlabel('')
        axs[2].set_xlabel('')
        axs[3].set_xlabel('DBSCAN River Type', fontsize=25)
        axs[4].set_xlabel('')
        axs[5].set_xlabel('')
        axs[6].set_xlabel('')

        #save prior distribution parameters
        priorWbClass = dbscan_df.groupby('cluster')['logWb'].describe()
        priorDbClass = dbscan_df.groupby('cluster')['logDb'].describe()
        prior_rClass = dbscan_df.groupby('cluster')['logr'].describe()
        priorA0Class = dbscan_df.groupby('cluster')['logA0'].describe()
        priorNClass = dbscan_df.groupby('cluster')['logn'].describe()
        priorBClass = dbscan_df.groupby('cluster')['b'].describe()

        self.DBSCANFolder = self.set_Folder(self.mergedFolder, "DBSCAN")
        self.judge_clustering_result(self.dataDF, pca_df["cluster"], self.DBSCANFolder, "DBSCAN")

        priorWbClass.to_csv(os.path.join(self.DBSCANFolder, "Wb_DBSCAN.csv")

        priorDbClass.to_csv(os.path.join(self.DBSCANFolder, "Db_DBSCAN.csv"))
        prior_rClass.to_csv(os.path.join(self.DBSCANFolder, "r_DBSCAN.csv"))
        priorA0Class.to_csv(os.path.join(self.DBSCANFolder, "A0_DBSCAN.csv"))
        priorNClass.to_csv(os.path.join(self.DBSCANFolder, "N_DBSCAN.csv"))
        priorBClass.to_csv(os.path.join(self.DBSCANFolder, "B_DBSCAN.csv"))

        fig.savefig(os.path.join(self.DBSCANFolder, "DBSCAN.png", dpi = 300))

        return 
    
    def PCA_classification(self):
        pca_df = self.dataDF
        #run PCA on these features
        features = ['chan_width', 'n', 'SLOPE', 'StreamOrde','DistDwnstrm', 'FCODEnorm', 'chan_depth', 'chan_velocity', 
                    'unitPower', 'r', 'DASqKm', 'Fb', 'shearStress', 'minEntrain', 'TOTMA', 'sinuosity', 'velocity_var',
                    'depth_var','width_var', 'n_var','unitPower_var','Fb_var','shearStress_var', 'minEntrain_var']#, 
                    #'CAT_SILTAVE', 'CAT_SANDAVE', 'CAT_CLAYAVE']
        x = pca_df.loc[:, features].values # Separating out the features
        #y = temp.loc[:,['site_no']].values # Separating out the target
        x = StandardScaler().fit_transform(x) # normalizing the features

        pca = PCA(n_components=3) #ran using 3 PCs
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data = pca.components_
                     , columns = ['chan_width', 'n', 'SLOPE', 'StreamOrde','DistDwnstrm', 'FCODEnorm', 'chan_depth', 'chan_velocity', 
                    'unitPower', 'r', 'DASqKm', 'Fb', 'shearStress', 'minEntrain', 'TOTMA', 'sinuosity', 'velocity_var',
                    'depth_var','width_var', 'n_var','unitPower_var','Fb_var','shearStress_var', 'minEntrain_var'])#,
                    #'CAT_SILTAVE', 'CAT_SANDAVE', 'CAT_CLAYAVE'])
        pca = PCA().fit(x)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance (%)')
        plt.title('Explained Variance vs. # Principal Components')

        principalDf.transpose()
        pca.explained_variance_ratio_[0:8]

        plt.show()
        #15 classes is the largest number before class median widths overlap, 14 part in average.
        quantiles = [0.067, 0.134, 0.201, 0.268, 0.335, 0.402, 0.469, 0.536, 0.603, 0.670, 0.737, 0.804, 0.871, 0.938]

        #add PC values to self.mergedData
        pca_df['PC1'] = principalComponents[:,0]
        pca_df['PC2'] = principalComponents[:,1]
        pca_df['PC3'] = principalComponents[:,2]
        #pca_df['PC4'] = principalComponents[:,3]

        pca_df['geomorphIndex'] = (pca_df['PC1'])+(pca_df['PC2'])+(pca_df['PC3'])#+(pca_df['PC4'])
        geomorphIndex = np.quantile(pca_df['geomorphIndex'], quantiles)

        pca_df['cluster'] = np.where(pca_df['logr']<0, '16',
                                                  np.where(pca_df['geomorphIndex']<geomorphIndex[0], '1', 
                                                       np.where(pca_df['geomorphIndex']<geomorphIndex[1], '2', 
                                                               np.where(pca_df['geomorphIndex']<geomorphIndex[2], '3', 
                                                                       np.where(pca_df['geomorphIndex']<geomorphIndex[3], '4',
                                                                                np.where(pca_df['geomorphIndex']<geomorphIndex[4], '5',
                                                                                         np.where(pca_df['geomorphIndex']<geomorphIndex[5], '6',
                                                                                                  np.where(pca_df['geomorphIndex']<geomorphIndex[6], '7', 
                                                                                                           np.where(pca_df['geomorphIndex']<geomorphIndex[7], '8',
                                                                                                                    np.where(pca_df['geomorphIndex']<geomorphIndex[8], '9',
                                                                                                                            np.where(pca_df['geomorphIndex']<geomorphIndex[9], '10',
                                                                                                                                     np.where(pca_df['geomorphIndex']<geomorphIndex[10], '11',
                                                                                                                                              np.where(pca_df['geomorphIndex']<geomorphIndex[11], '12',
                                                                                                                                                       np.where(pca_df['geomorphIndex']<geomorphIndex[12], '13',
                                                                                                                                                           np.where(pca_df['geomorphIndex']<geomorphIndex[13], '14','15')))))))))))))))
        self.PCAFolder = self.set_Folder(self.mergedFolder, "PCA")
        self.judge_clustering_result(pca_df, pca_df["cluster"], self.PCAFolder, "PCA")

        temp = pca_df[['logchan_width', 'cluster']]

        pca_df['logchan_width'] = np.log(pca_df['chan_width']) #mean at-a-station width

        fig, axs = plt.subplots(ncols=7, figsize=(20, 7))
        sns.boxplot(x="cluster", y='logA0', data=pca_df, palette='deep', ax=axs[0], order=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'])
        sns.boxplot(x="cluster", y='logn', data=pca_df, palette='deep', ax=axs[1], order=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'])
        sns.boxplot(x="cluster", y='b', data=pca_df, palette='deep', ax=axs[2], order=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'])
        sns.boxplot(x="cluster", y='logWb', data=pca_df, palette='deep', ax=axs[3], order=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'])
        sns.boxplot(x="cluster", y='logDb', data=pca_df, palette='deep', ax=axs[4], order=['1', '2', '3', '4 ', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'])
        sns.boxplot(x="cluster", y='logr', data=pca_df, palette='deep', ax=axs[5], order=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'])
        sns.boxplot(x="cluster", y='logchan_width', data=pca_df, palette='deep', ax=axs[6], order=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'])

        fig.autofmt_xdate(rotation=90)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        axs[0].set_ylabel('log A0', fontsize=15)
        axs[1].set_ylabel('log Mannings n', fontsize=15)
        axs[2].set_ylabel('b', fontsize=15)
        axs[3].set_ylabel('log Wb', fontsize=15)
        axs[4].set_ylabel('log Db', fontsize=15)
        axs[5].set_ylabel('log r', fontsize=15)
        axs[6].set_ylabel('log W', fontsize=15)

        axs[5].set_ylim(-1.5,2.5)

        axs[0].set_xlabel('')
        axs[1].set_xlabel('')
        axs[2].set_xlabel('')
        axs[3].set_xlabel('Expert River Types', fontsize=25)
        axs[4].set_xlabel('')
        axs[5].set_xlabel('')
        axs[6].set_xlabel('')

        priorWbClass = pca_df.groupby('cluster')['logWb'].describe()
        priorDbClass = pca_df.groupby('cluster')['logDb'].describe()
        prior_rClass = pca_df.groupby('cluster')['logr'].describe()
        priorA0Class = pca_df.groupby('cluster')['logA0'].describe()
        priorNClass = pca_df.groupby('cluster')['logn'].describe()
        priorBClass = pca_df.groupby('cluster')['b'].describe()

        priorWbClass.to_csv(os.path.join(self.PCAFolder, "Wb_pca.csv"))
        priorDbClass.to_csv(os.path.join(self.PCAFolder, "Db_pca.csv"))
        prior_rClass.to_csv(os.path.join(self.PCAFolder, "r_pca.csv"))
        priorA0Class.to_csv(os.path.join(self.PCAFolder, "A0_pca.csv"))
        priorNClass.to_csv(os.path.join(self.PCAFolder, "N_pca.csv"))
        priorBClass.to_csv(os.path.join(self.PCAFolder, "B_pca.csv"))

        fig.savefig(os.path.join((self.PCAFolder, "PCA.png"), dpi = 300)

    def judge_clustering_result(self, df, Folder, name):

        ss = silhouette_score(df, df["cluster"])
        chs = calinski_harabasz_score(df, df["cluster"])
        dbs = davies_bouldin_score(df, df["cluster"])
        pd.DataFrame([ss,chs,dbs]).to_csv(os.path.join(Folder, name + '_metric.csv'))

    def write_result_to_new(self):
        return 

    def run_all(self):
        if not os.path.exists(self.mergedDataFile):
            self.merge_NWIS_NHD()

        if not os.path.exists(self.NHDLakeFile):
            self.read_NHD_lake()

        self.select_and_cal_valuable()
        
        self.DBSCAN_classification()
        self.PCA_classification()





def main():
    
    log = logging.getLogger()
    handler = logging.StreamHandler()
    log.addHandler(handler)
    log.setLevel(logging.INFO)

    workspace = r"D:\RiverDischargeData"

    datasetCreater = DatasetCreater(workspace)
    datasetCreater.run_all()
     
if __name__ == '__main__':
    main() 
