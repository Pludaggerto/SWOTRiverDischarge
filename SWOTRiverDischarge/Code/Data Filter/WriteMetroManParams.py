import logging
import glob
import os 
import netCDF4 as nc
import shutil
import numpy as np

class ParamsWriter(object):

    def __init__(self, workspace, name):
        logging.info("[INFO]Writing params begins...")
        self.workspace = workspace
        self.fileName = os.path.join(self.workspace, "Brahmaputra.nc")
        self.name = name
        self.templateFolder = os.path.join(os.path.dirname(__file__), "template")
        self.ReachLength = 250000 # m

    def __del__(self):
        logging.info("[INFO]Writing params end...")

    def read_data(self):
        self.dataset = nc.Dataset(self.fileName)

        self.A = self.dataset["Reach_Timeseries/A"][:].transpose()
        self.S = self.dataset["Reach_Timeseries/S"][:].transpose()
        self.W = self.dataset["Reach_Timeseries/W"][:].transpose()
        self.H = self.dataset["Reach_Timeseries/H"][:].transpose()
        self.Q = self.dataset["Reach_Timeseries/Q"][:].transpose()

        self.dA = np.zeros_like(self.A)
        for i in range(self.dA.shape[1]):
            self.dA[:,i] = self.A[:,i] - self.A[:,0]

        return None

    def create_params_folder(self):

        self.targetFolder = os.path.join(self.workspace, self.name)

        if not os.path.exists(self.targetFolder):
            shutil.copytree(self.templateFolder, self.targetFolder)
        else:
            shutil.rmtree(self.targetFolder)
            shutil.copytree(self.templateFolder, self.targetFolder)
       
    def write_files(self):
        # params
        fParam = open(os.path.join(self.targetFolder, "params.txt"), "r")
        fParamLines = fParam.readlines()
        fParamLines[9] = str(self.Q.mean()) + "\t\n"
        fParam.close()
        fParam = open(os.path.join(self.targetFolder, "params.txt"), "w")
        fParam.writelines(fParamLines)
        fParam.close()
        
        # SWOTObs
        fSWOTobs = open(os.path.join(self.targetFolder, "SWOTObs.txt"), "r")
        fSWOTobsLines = fSWOTobs.readlines()

        # Number of reaches
        fSWOTobsLines[1] = str(self.A.shape[0]) + "\t\n"

        # Reach midpoint distance downstream, m
        midPointDistance = self.ReachLength / self.A.shape[0]
        midPointDistanceList = [str(i * midPointDistance) for i in range(self.A.shape[0])]
        midPointDistanceList = [str(i) for i in list(self.dataset["River_Info/rch_bnd"][:])[1:]]
        fSWOTobsLines[3] = "\t".join(midPointDistanceList) + "\n"

        # Reach lengths, m
        reachesLen = self.dataset["River_Info/rch_bnd"][1:] - self.dataset["River_Info/rch_bnd"][:-1]
        fSWOTobsLines[5] = "\t".join([str(i) for i in reachesLen]) + "\n"

        # Time, days
        fSWOTobsLines[7] = str(int(self.A.shape[1])) + "\n"

        dayList = [str(i) for i in range(self.A.shape[1])]
        fSWOTobsLines[9] = "\t".join(dayList) + "\n"
        fSWOTobsLines = fSWOTobsLines[0:10]

        ## Height, meters\n
        self.write_name_and_matrix(fSWOTobsLines, "Height, meters\n", self.H)

        ## Height at baseflow, m
        fSWOTobsLines.append("Height at baseflow, m\n")
        line = "\t".join([str(j) for j in list(self.H[:,0])]) + "\n"
        fSWOTobsLines.append(line)

        ## Slope, cm/km
        self.write_name_and_matrix(fSWOTobsLines, "Slope, cm/km\n", self.S * 100000) #!!! change unit

        ## Width, m
        self.write_name_and_matrix(fSWOTobsLines, "Width, m\n", self.W)

        fSWOTobsLines.append("Standard deviation on slope cm/km \n")
        fSWOTobsLines.append("0.1000 \n")
        fSWOTobsLines.append("Standard deviation on height cm  \n")
        fSWOTobsLines.append("0.1000 \n")
        fSWOTobsLines.append("Standard deviation on width m  \n")
        fSWOTobsLines.append("0.1000 \n")

        fSWOTobs.close()
        fSWOTobs = open(os.path.join(self.targetFolder, "SWOTObs.txt"), "w")
        fSWOTobs.writelines(fSWOTobsLines)
        fSWOTobs.close()

        # truth
        fTruth = open(os.path.join(self.targetFolder, "truth.txt"), "r")
        fTruthLines = fTruth.readlines()

        line = [str(j) for j in list(self.A[:,0])]
        fTruthLines[1] = "\t".join(line) + "\n"
        fTruthLines = fTruthLines[0:6]

        ## Qtrue [m3/s]
        self.write_name_and_matrix(fTruthLines, "Qtrue [m3/s]\n", self.Q)

        ## dA, m2
        self.write_name_and_matrix(fTruthLines, "dA, m2\n", self.dA)

        # h, m
        self.write_name_and_matrix(fTruthLines, "h, m\n", self.H)

        # W, m
        self.write_name_and_matrix(fTruthLines, "W, m\n", self.W)

        fTruth.close()
        fTruth = open(os.path.join(self.targetFolder, "truth.txt"), "w")
        fTruth.writelines(fTruthLines)
        fTruth.close()

    def write_name_and_matrix(self, WriteLines, name, matrix):
        WriteLines.append(name)
        for i in list(matrix):
            line = [str(j) for j in list(i)]
            WriteLines.append("\t".join(list(line)) + "\n")

    def copy_files(self):
        resultFolder = os.path.join(r"C:\Users\lwx\source\repos\RiverDishcharge\RiverDishcharge\Code\MetroMan-master", self.name)

        if not os.path.exists(resultFolder):
            shutil.copytree(self.targetFolder, resultFolder)
        else:
            shutil.rmtree(resultFolder)
            shutil.copytree(self.targetFolder, resultFolder)

    def run_all(self):
        self.create_params_folder()
        self.read_data()
        self.write_files()
        self.copy_files()

def main():
    
    log = logging.getLogger()
    handler = logging.StreamHandler()
    log.addHandler(handler)
    log.setLevel(logging.INFO)

    workspace = r"C:\Users\lwx\Desktop\Dishcarge\Ideal-Data"
    name = "Brahmaputra"

    paramsWriter = ParamsWriter(workspace, name)
    paramsWriter.run_all()
     
if __name__ == '__main__':
    main() 