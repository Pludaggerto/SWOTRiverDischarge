import dataretrieval.nwis as nwis
import pandas             as pd
import geopandas          as gpd
import matplotlib.pyplot  as plt
import numpy              as np
import scipy.stats.mstats as sp
import seaborn            as sns
import statsmodels.api    as sm

import logging
import glob
import os 

from dbfread                 import DBF
from sklearn                 import metrics
from sklearn                 import svm
from sklearn                 import linear_model
from sklearn.cluster         import DBSCAN, KMeans
from sklearn.preprocessing   import StandardScaler, normalize
from sklearn.decomposition   import PCA
from sklearn.metrics         import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors       import NearestNeighbors

class Merger(object):

    def __init__(self):
        return 

    def __del__(self):
        return

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
        self.mergedData.to_csv(self.mergedCaledDataFile, index = False)

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
        self.dataDF.to_csv(self.FeatureFile, index = False)


