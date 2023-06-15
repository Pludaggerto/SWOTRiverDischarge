import dataretrieval.nwis as nwis
import pandas             as pd
import geopandas          as gpd
import matplotlib.pyplot  as plt
import numpy              as np
import scipy.stats.mstats as sp
import seaborn            as sns
import statsmodels.api    as sm

import os 

from dbfread                 import DBF
from sklearn                 import metrics
from sklearn                 import svm
from sklearn                 import linear_model
from sklearn.cluster         import DBSCAN, AffinityPropagation, MeanShift, OPTICS, Birch
from sklearn.preprocessing   import StandardScaler, normalize
from sklearn.decomposition   import PCA
from sklearn.metrics         import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors       import NearestNeighbors

class Clusterer(object):

    def __init__(self):
        return 

    def __del__(self):
        return

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

    def DBSCAN_cluster(self):
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
        for eps in np.arange(0.1, 10, 0.1):
            for min_samples in np.arange(1, 100, 5):
                for leaf_size in np.arange(20, 200, 20):
                    clustering = DBSCAN(eps = eps, min_samples = min_samples, leaf_size = leaf_size).fit(X_normalized)
                    labels = clustering.labels_
                    dataDF["cluster"] = labels
                    self.judge_clustering_result(dataDF, self.mergedFolder, "DBSCAN", "eps:" + str(eps) + "_" "min_samples:" + "_" + str(min_samples) + "leaf_size:" + "_" + str(leaf_size))

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)

        sns.set_style("white")
        dataDF = self.dataDF 
        fig, axs = plt.subplots(ncols=7, figsize=(20, 7))
        dataDF["cluster"] = labels
        sns.boxplot(x="cluster", y='logA0', data=dataDF, palette='deep', ax=axs[0])
        sns.boxplot(x="cluster", y='logn', data=dataDF, palette='deep', ax=axs[1])
        sns.boxplot(x="cluster", y='b', data=dataDF, palette='deep', ax=axs[2])
        sns.boxplot(x="cluster", y='logWb', data=dataDF, palette='deep', ax=axs[3])
        sns.boxplot(x="cluster", y='logDb', data=dataDF, palette='deep', ax=axs[4])
        sns.boxplot(x="cluster", y='logr', data=dataDF, palette='deep', ax=axs[5])
        sns.boxplot(x="cluster", y='logchan_width', data=dataDF, palette='deep', ax=axs[6])

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
        priorWbClass = dataDF.groupby('cluster')['logWb'].describe()
        priorDbClass = dataDF.groupby('cluster')['logDb'].describe()
        prior_rClass = dataDF.groupby('cluster')['logr'].describe()
        priorA0Class = dataDF.groupby('cluster')['logA0'].describe()
        priorNClass = dataDF.groupby('cluster')['logn'].describe()
        priorBClass = dataDF.groupby('cluster')['b'].describe()

        priorWbClass.to_csv(os.path.join(self.DBSCANFolder, "Wb_DBSCAN.csv"))

        priorDbClass.to_csv(os.path.join(self.DBSCANFolder, "Db_DBSCAN.csv"))
        prior_rClass.to_csv(os.path.join(self.DBSCANFolder, "r_DBSCAN.csv"))
        priorA0Class.to_csv(os.path.join(self.DBSCANFolder, "A0_DBSCAN.csv"))
        priorNClass.to_csv(os.path.join(self.DBSCANFolder, "N_DBSCAN.csv"))
        priorBClass.to_csv(os.path.join(self.DBSCANFolder, "B_DBSCAN.csv"))

        fig.savefig(os.path.join(self.DBSCANFolder, "DBSCAN.png"), dpi = 300)

        return 
    
    def PCA_cluster(self):
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
        fig, axs = plt.subplots(1, 1 , figsize=(20, 7))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance (%)')
        plt.title('Explained Variance vs. # Principal Components')

        principalDf.transpose()
        pca.explained_variance_ratio_[0:8]

        fig.savefig(os.path.join(self.PCAFolder, "PCA_components.png"), dpi = 300)
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
        
        self.judge_clustering_result(pca_df, self.mergedFolder, "PCA")

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

        fig.savefig(os.path.join(self.PCAFolder, "PCA.png"), dpi = 300)       

    def AffinityPropagation_cluster(self):  
        X_normalized = self.X_normalized
        AP_df = self.dataDF
        #run PCA on these features
        features = ['chan_width', 'n', 'SLOPE', 'StreamOrde','DistDwnstrm', 'FCODEnorm', 'chan_depth', 'chan_velocity', 
                    'unitPower', 'r', 'DASqKm', 'Fb', 'shearStress', 'minEntrain', 'TOTMA', 'sinuosity', 'velocity_var',
                    'depth_var','width_var', 'n_var','unitPower_var','Fb_var','shearStress_var', 'minEntrain_var']#, 
                    #'CAT_SILTAVE', 'CAT_SANDAVE', 'CAT_CLAYAVE']
        x = AP_df.loc[:, features] # Separating out the features

        for damping in np.arange(0.5,1,0.3):
            clustering = AffinityPropagation(damping = damping).fit(self.X_normalized)
            labels = clustering.labels_
            AP_df["cluster"] = labels
            self.judge_clustering_result(AP_df, self.mergedFolder, "AffPro", "damping:" + str(damping))

        
        labels = clustering.labels_
        sns.set_style("white")
        dataDF = self.dataDF 
        fig, axs = plt.subplots(ncols=7, figsize=(20, 7))
        dataDF["cluster"] = labels
        sns.boxplot(x="cluster", y='logA0', data=dataDF, palette='deep', ax=axs[0])
        sns.boxplot(x="cluster", y='logn', data=dataDF, palette='deep', ax=axs[1])
        sns.boxplot(x="cluster", y='b', data=dataDF, palette='deep', ax=axs[2])
        sns.boxplot(x="cluster", y='logWb', data=dataDF, palette='deep', ax=axs[3])
        sns.boxplot(x="cluster", y='logDb', data=dataDF, palette='deep', ax=axs[4])
        sns.boxplot(x="cluster", y='logr', data=dataDF, palette='deep', ax=axs[5])
        sns.boxplot(x="cluster", y='logchan_width', data=dataDF, palette='deep', ax=axs[6])

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
        axs[3].set_xlabel('AffPro River Type', fontsize=25)
        axs[4].set_xlabel('')
        axs[5].set_xlabel('')
        axs[6].set_xlabel('')

        #save prior distribution parameters
        priorWbClass = dataDF.groupby('cluster')['logWb'].describe()
        priorDbClass = dataDF.groupby('cluster')['logDb'].describe()
        prior_rClass = dataDF.groupby('cluster')['logr'].describe()
        priorA0Class = dataDF.groupby('cluster')['logA0'].describe()
        priorNClass = dataDF.groupby('cluster')['logn'].describe()
        priorBClass = dataDF.groupby('cluster')['b'].describe()

        priorWbClass.to_csv(os.path.join(self.AffinityPropagationFolder, "Wb_AffPro.csv"))

        priorDbClass.to_csv(os.path.join(self.AffinityPropagationFolder, "Db_AffPro.csv"))
        prior_rClass.to_csv(os.path.join(self.AffinityPropagationFolder, "r_AffPro.csv"))
        priorA0Class.to_csv(os.path.join(self.AffinityPropagationFolder, "A0_AffPro.csv"))
        priorNClass.to_csv(os.path.join(self.AffinityPropagationFolder, "N_AffPro.csv"))
        priorBClass.to_csv(os.path.join(self.AffinityPropagationFolder, "B_AffPro.csv"))

        fig.savefig(os.path.join(self.AffinityPropagationFolder, "AffPro.png"), dpi = 300)
        return

    def meanShift_cluster(self):
        meanshift_df = self.dataDF
        #run PCA on these features
        features = ['chan_width', 'n', 'SLOPE', 'StreamOrde','DistDwnstrm', 'FCODEnorm', 'chan_depth', 'chan_velocity', 
                    'unitPower', 'r', 'DASqKm', 'Fb', 'shearStress', 'minEntrain', 'TOTMA', 'sinuosity', 'velocity_var',
                    'depth_var','width_var', 'n_var','unitPower_var','Fb_var','shearStress_var', 'minEntrain_var']#, 
                    #'CAT_SILTAVE', 'CAT_SANDAVE', 'CAT_CLAYAVE']
        x = meanshift_df.loc[:, features].values # Separating out the features
        clustering = MeanShift().fit(x)
        labels = clustering.labels_
        sns.set_style("white")
        dataDF = self.dataDF 
        fig, axs = plt.subplots(ncols=7, figsize=(20, 7))
        dataDF["cluster"] = labels
        sns.boxplot(x="cluster", y='logA0', data=dataDF, palette='deep', ax=axs[0])
        sns.boxplot(x="cluster", y='logn', data=dataDF, palette='deep', ax=axs[1])
        sns.boxplot(x="cluster", y='b', data=dataDF, palette='deep', ax=axs[2])
        sns.boxplot(x="cluster", y='logWb', data=dataDF, palette='deep', ax=axs[3])
        sns.boxplot(x="cluster", y='logDb', data=dataDF, palette='deep', ax=axs[4])
        sns.boxplot(x="cluster", y='logr', data=dataDF, palette='deep', ax=axs[5])
        sns.boxplot(x="cluster", y='logchan_width', data=dataDF, palette='deep', ax=axs[6])

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
        axs[3].set_xlabel('AffPro River Type', fontsize=25)
        axs[4].set_xlabel('')
        axs[5].set_xlabel('')
        axs[6].set_xlabel('')

        #save prior distribution parameters
        priorWbClass = dataDF.groupby('cluster')['logWb'].describe()
        priorDbClass = dataDF.groupby('cluster')['logDb'].describe()
        prior_rClass = dataDF.groupby('cluster')['logr'].describe()
        priorA0Class = dataDF.groupby('cluster')['logA0'].describe()
        priorNClass = dataDF.groupby('cluster')['logn'].describe()
        priorBClass = dataDF.groupby('cluster')['b'].describe()

        
        self.judge_clustering_result(dataDF, self.mergedFolder, "meanshift")

        priorWbClass.to_csv(os.path.join(self.meanShiftFolder, "Wb_meanshift.csv"))

        priorDbClass.to_csv(os.path.join(self.meanShiftFolder, "Db_meanshift.csv"))
        prior_rClass.to_csv(os.path.join(self.meanShiftFolder, "r_meanshift.csv"))
        priorA0Class.to_csv(os.path.join(self.meanShiftFolder, "A0_meanshift.csv"))
        priorNClass.to_csv(os.path.join(self.meanShiftFolder, "N_meanshift.csv"))
        priorBClass.to_csv(os.path.join(self.meanShiftFolder, "B_meanshift.csv"))

        fig.savefig(os.path.join(self.AffinityPropagationFolder, "meanshift.png"), dpi = 300)
        return

    def OPTICS_cluster(self):
        OPTICS_df = self.dataDF
        #run PCA on these features
        features = ['chan_width', 'n', 'SLOPE', 'StreamOrde','DistDwnstrm', 'FCODEnorm', 'chan_depth', 'chan_velocity', 
                    'unitPower', 'r', 'DASqKm', 'Fb', 'shearStress', 'minEntrain', 'TOTMA', 'sinuosity', 'velocity_var',
                    'depth_var','width_var', 'n_var','unitPower_var','Fb_var','shearStress_var', 'minEntrain_var']#, 
                    #'CAT_SILTAVE', 'CAT_SANDAVE', 'CAT_CLAYAVE']
        x = OPTICS_df.loc[:, features].values # Separating out the features
        clustering = OPTICS().fit(self.X_normalized)
        labels = clustering.labels_
        sns.set_style("white")
        dataDF = self.dataDF 
        fig, axs = plt.subplots(ncols=7, figsize=(20, 7))
        dataDF["cluster"] = labels
        sns.boxplot(x="cluster", y='logA0', data=dataDF, palette='deep', ax=axs[0])
        sns.boxplot(x="cluster", y='logn', data=dataDF, palette='deep', ax=axs[1])
        sns.boxplot(x="cluster", y='b', data=dataDF, palette='deep', ax=axs[2])
        sns.boxplot(x="cluster", y='logWb', data=dataDF, palette='deep', ax=axs[3])
        sns.boxplot(x="cluster", y='logDb', data=dataDF, palette='deep', ax=axs[4])
        sns.boxplot(x="cluster", y='logr', data=dataDF, palette='deep', ax=axs[5])
        sns.boxplot(x="cluster", y='logchan_width', data=dataDF, palette='deep', ax=axs[6])

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
        axs[3].set_xlabel('AffPro River Type', fontsize=25)
        axs[4].set_xlabel('')
        axs[5].set_xlabel('')
        axs[6].set_xlabel('')

        #save prior distribution parameters
        priorWbClass = dataDF.groupby('cluster')['logWb'].describe()
        priorDbClass = dataDF.groupby('cluster')['logDb'].describe()
        prior_rClass = dataDF.groupby('cluster')['logr'].describe()
        priorA0Class = dataDF.groupby('cluster')['logA0'].describe()
        priorNClass = dataDF.groupby('cluster')['logn'].describe()
        priorBClass = dataDF.groupby('cluster')['b'].describe()

        
        self.judge_clustering_result(dataDF, self.mergedFolder, "OPTICS")

        priorWbClass.to_csv(os.path.join(self.OPTICSFolder, "Wb_OPTICS.csv"))

        priorDbClass.to_csv(os.path.join(self.OPTICSFolder, "Db_OPTICS.csv"))
        prior_rClass.to_csv(os.path.join(self.OPTICSFolder, "r_OPTICS.csv"))
        priorA0Class.to_csv(os.path.join(self.OPTICSFolder, "A0_OPTICS.csv"))
        priorNClass.to_csv(os.path.join(self.OPTICSFolder, "N_OPTICS.csv"))
        priorBClass.to_csv(os.path.join(self.OPTICSFolder, "B_OPTICS.csv"))

        fig.savefig(os.path.join(self.AffinityPropagationFolder, "OPTICS.png"), dpi = 300)
        return

    def BIRCH_cluster(self):

        OPTICS_df = self.dataDF
        #run PCA on these features
        features = ['chan_width', 'n', 'SLOPE', 'StreamOrde','DistDwnstrm', 'FCODEnorm', 'chan_depth', 'chan_velocity', 
                    'unitPower', 'r', 'DASqKm', 'Fb', 'shearStress', 'minEntrain', 'TOTMA', 'sinuosity', 'velocity_var',
                    'depth_var','width_var', 'n_var','unitPower_var','Fb_var','shearStress_var', 'minEntrain_var']#, 
                    #'CAT_SILTAVE', 'CAT_SANDAVE', 'CAT_CLAYAVE']
        x = OPTICS_df.loc[:, features].values # Separating out the features
        for n_clusters in np.arange(1,30,1):
            for branching_factor in np.arange(30, 70,5):
                clustering = Birch(n_clusters = n_clusters, branching_factor = branching_factor).fit(self.X_normalized)                    
                labels = clustering.labels_
                OPTICS_df["cluster"] = labels
                self.judge_clustering_result(OPTICS_df, self.mergedFolder, "BIRCH", "n_clusters:" + str(n_clusters) + "_" "branching_factor:" + "_" + str(branching_factor))
        
        labels = clustering.labels_
        sns.set_style("white")
        dataDF = self.dataDF 
        fig, axs = plt.subplots(ncols=7, figsize=(20, 7))
        dataDF["cluster"] = labels
        sns.boxplot(x="cluster", y='logA0', data=dataDF, palette='deep', ax=axs[0])
        sns.boxplot(x="cluster", y='logn', data=dataDF, palette='deep', ax=axs[1])
        sns.boxplot(x="cluster", y='b', data=dataDF, palette='deep', ax=axs[2])
        sns.boxplot(x="cluster", y='logWb', data=dataDF, palette='deep', ax=axs[3])
        sns.boxplot(x="cluster", y='logDb', data=dataDF, palette='deep', ax=axs[4])
        sns.boxplot(x="cluster", y='logr', data=dataDF, palette='deep', ax=axs[5])
        sns.boxplot(x="cluster", y='logchan_width', data=dataDF, palette='deep', ax=axs[6])

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
        axs[3].set_xlabel('AffPro River Type', fontsize=25)
        axs[4].set_xlabel('')
        axs[5].set_xlabel('')
        axs[6].set_xlabel('')

        #save prior distribution parameters
        priorWbClass = dataDF.groupby('cluster')['logWb'].describe()
        priorDbClass = dataDF.groupby('cluster')['logDb'].describe()
        prior_rClass = dataDF.groupby('cluster')['logr'].describe()
        priorA0Class = dataDF.groupby('cluster')['logA0'].describe()
        priorNClass = dataDF.groupby('cluster')['logn'].describe()
        priorBClass = dataDF.groupby('cluster')['b'].describe()

        
        self.judge_clustering_result(dataDF, self.mergedFolder, "Birch")

        priorWbClass.to_csv(os.path.join(self.BIRCHFolder, "Wb_Birch.csv"))

        priorDbClass.to_csv(os.path.join(self.BIRCHFolder, "Db_Birch.csv"))
        prior_rClass.to_csv(os.path.join(self.BIRCHFolder, "r_Birch.csv"))
        priorA0Class.to_csv(os.path.join(self.BIRCHFolder, "A0_Birch.csv"))
        priorNClass.to_csv(os.path.join(self.BIRCHFolder, "N_Birch.csv"))
        priorBClass.to_csv(os.path.join(self.BIRCHFolder, "B_Birch.csv"))

        fig.savefig(os.path.join(self.AffinityPropagationFolder, "Birch.png"), dpi = 300)
        return

    def judge_clustering_result(self, df, folder, name, params = ''):

        cluster = df["cluster"]
        dftemp = df.drop(['cluster'], axis = 1)

        # Silhouette Coefficient: si接近1，则说明样本i聚类合理
        try:
            ss = silhouette_score(dftemp, cluster)

        except:
            ss = -1

         # Calinski-Harabaz Index:分数值ss越大则聚类效果越好
        try:
            chs = calinski_harabasz_score(dftemp, cluster)

        except:
            chs = -1

         # DBI(davies_bouldin_score)：值最小是0，值越小，代表聚类效果越好。
        try:
            dbs = davies_bouldin_score(dftemp, cluster)

        except:
            dbs = -1
           
            
        result = pd.DataFrame([ss, chs, dbs]).transpose()
        result.columns = ["ss", "chs", "dbs"]
        path = os.path.join(folder, "result.csv")
        result["name"] = name + "_" + params
        result["group"] = cluster.astype(int).max()

        if cluster.astype(int).max() < 2:
            return

        if os.path.exists(path):
            result.to_csv(path, mode = "a", header = None, index = False)
        else:
            result.to_csv(path, mode = "a", index = False)
        return ss, chs, dbs